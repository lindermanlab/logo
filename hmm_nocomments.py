from functools import partial
from tqdm.auto import trange
from textwrap import dedent
from time import time_ns

import jax.numpy as np
import jax.random as jr
from jax import jit, lax
from jax.tree_util import register_pytree_node, register_pytree_node_class

import jxf.distributions as dists

from ssm.hmm.initial_state import InitialState, UniformInitialState
from ssm.hmm.transitions import Transitions, StationaryTransitions
from ssm.hmm.observations import Observations
from ssm.hmm.posteriors import HMMPosterior
from ssm.util import ssm_pbar, format_dataset, num_datapoints, Verbosity


@register_pytree_node_class
class HMM(object):
    def __init__(self, num_states,
                 initial_state="uniform",
                 initial_state_kwargs={},
                 transitions="standard",
                 transitions_prior=None,
                 transition_kwargs={},
                 observations="gaussian",
                 observations_prior=None,
                 observation_kwargs={}):
        self.num_states = num_states
        self.initial_state = self.__build_initial_state(
            num_states, initial_state, **initial_state_kwargs)
        self.transitions = self.__build_transitions(
            num_states, transitions, transitions_prior, **transition_kwargs)
        self.observations = self.__build_observations(
            num_states, observations, observations_prior, **observation_kwargs)

    def tree_flatten(self):
        return ((self.initial_state,
                 self.transitions,
                 self.observations), self.num_states)

    @classmethod
    def tree_unflatten(cls, num_states, children):
        initial_state, transitions, observations = children
        return cls(num_states,
                   initial_state=initial_state,
                   transitions=transitions,
                   observations=observations)

    def __build_initial_state(self, num_states,
                              initial_state,
                              **initial_state_kwargs):
        initial_state_names = dict(
            uniform=UniformInitialState,
        )
        if isinstance(initial_state, str):
            return initial_state_names[initial_state.lower()](num_states, **initial_state_kwargs)
        else:
            assert isinstance(initial_state, InitialState)
            return initial_state

    def __build_transitions(self, num_states,
                            transitions,
                            transitions_prior,
                            **transitions_kwargs):
        if isinstance(transitions, np.ndarray):
            # Assume this is a transition matrix
            return StationaryTransitions(num_states,
                                        transition_matrix=transitions)

        elif isinstance(transitions, str):
            # String specifies class of transitions
            transition_class = _TRANSITION_CLASSES[transitions.lower()]
            return transition_class(num_states, **transitions_kwargs)
        else:
            # Otherwise, we need a Transitions object
            return transitions


    @property
    def transition_matrix(self):
        return self.transitions.get_transition_matrix()

    @transition_matrix.setter
    def transition_matrix(self, value):
        return self.transitions.set_transition_matrix(value)

    @property
    def observation_distributions(self):
        return self.observations.conditional_dists

    @format_dataset
    def initialize(self, rng, dataset):
        keys = jr.split(rng, 3)
        components = [self.initial_state, self.transitions, self.observations]
        for key, component in zip(keys, components):
            component.initialize(key, dataset)

    def permute(self, perm):
        self.initial_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def log_prior(self):
        return self.initial_state.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @format_dataset
    def average_log_likelihood(self, dataset):
        posteriors = [HMMPosterior(self, data_dict) for data_dict in dataset]
        lp = np.sum([p.marginal_likelihood() for p in posteriors])
        return lp / num_datapoints(dataset), posteriors

    @format_dataset
    def average_log_prob(self, dataset):
        posteriors = [HMMPosterior(self, data_dict) for data_dict in dataset]
        lp = self.log_prior()
        lp += np.sum([p.marginal_likelihood() for p in posteriors])
        return lp / num_datapoints(dataset), posteriors

    def sample(self, rng, num_timesteps, prefix=None, covariates=None, **kwargs):
        rng_init, rng = jr.split(rng, 2)
        initial_state = jr.choice(rng_init, self.num_states)

        # Precompute sample functions for each observation and transition distribution
        def _sample(d): return lambda seed: d.sample(seed=seed)
        trans_sample_funcs = [_sample(d) for d in self.transitions.conditional_dists]
        obs_sample_funcs = [_sample(d) for d in self.observations.conditional_dists]

        # Sample one step at a time with lax.scan
        keys = jr.split(rng, num_timesteps)
        def sample_next(curr_state, key):
            key1, key2 = jr.split(key, 2)

            # Sample observation
            curr_obs = lax.switch(curr_state, obs_sample_funcs, key1)

            # Sample next state
            next_state = lax.switch(curr_state, trans_sample_funcs, key2)
            return next_state, (curr_state, curr_obs)

        _, (states, data) = lax.scan(sample_next, initial_state, keys)
        return states, data


    @format_dataset
    def infer_posterior(self, dataset):
        posteriors = [HMMPosterior(self, data) for data in dataset]
        return posteriors[0] if len(posteriors) == 1 else posteriors

    @format_dataset
    def _fit_em(self, rng, dataset, num_iters=100, tol=1e-4, verbosity=Verbosity.LOUD):
        @jit
        def step(model):
            # E Step
            posteriors = [HMMPosterior(model, data) for data in dataset]

            # Compute log probability
            lp = model.log_prior()
            lp += sum([p.marginal_likelihood() for p in posteriors])

            # M Step
            model.initial_state.m_step(dataset, posteriors)
            model.transitions.m_step(dataset, posteriors)
            model.observations.m_step(dataset, posteriors)
            return model, lp / num_datapoints(dataset)

        # Run the EM algorithm to convergence
        model = self
        log_probs = [np.nan]
        pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, log_probs[-1])
        for itr in pbar:
            model, lp = step(model)
            log_probs.append(lp)

            # Update progress bar
            if verbosity >= Verbosity.LOUD:
                pbar.set_description("LP: {:.3f}".format(lp))
                pbar.update(1)

            # Check for convergence
            if abs(log_probs[-1] - log_probs[-2]) < tol and itr > 1:
                break

        # Copy over the final model parameters
        self.initial_state = model.initial_state
        self.transitions = model.transitions
        self.observations = model.observations

        # Compute the posterior distribution(s) with the optimized parameters
        posteriors = [HMMPosterior(self, data) for data in dataset] \
            if len(dataset) > 1 else HMMPosterior(self, dataset[0])

        return np.array(log_probs), posteriors

    @format_dataset
    def fit(self,
            dataset,
            method="em",
            rng=None,
            initialize=True,
            verbose=Verbosity.LOUD,
            **kwargs):
        make_rng = (rng is None)
        if make_rng:
            rng = jr.PRNGKey(time_ns())

        _fitting_methods = dict(
            em=self._fit_em,
            # stochastic_em=self._fit_stochastic_em,
            )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        if initialize:
            # TODO: allow for kwargs to initialize
            rng_init, rng = jr.split(rng, 2)
            if verbose >= Verbosity.LOUD : print("Initializing...")
            self.initialize(rng_init, dataset)
            if verbose >= Verbosity.LOUD: print("Done.")

        # Run the fitting algorithm
        results = _fitting_methods[method](rng, dataset, **kwargs)
        return (rng, results) if make_rng else results