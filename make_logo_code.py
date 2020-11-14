import os
import pickle
from functools import partial
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point, LineString
from shapely.ops import unary_union
from shapely.affinity import translate, rotate, scale
from descartes.patch import PolygonPatch
from tqdm.auto import trange


# Stanford color palette at
# https://identity.stanford.edu/color.html#print-color
colors = dict(
    Ls="#F4F4F4",
    bkgd="#8c1515",
    chamber="#0098DB",
    sun="#2e2d29",
    planet="#2e2d29",
    wheel1="#2e2d29",
    wheel2="#2e2d29",
    piston="#2F2424",
    pad="#2e2d29",
    belt="#dad7cb",
    gmm="#F9F6EF",
    code="#928b81",
    kde="#928b81"
)

import ssm
TEXTFILE = "hmm_nocomments.py"


def _single_L(x, y, height=5.75, width=5, weight=1, serif=.75):
    """Draw a single block L"""
    boxes = []
    # Add the horizontal and vertical bars
    boxes.append(box(x+serif, y, x+serif+weight, y+height))
    boxes.append(box(x, y, x+width, y+weight))
    # Add the serifs
    boxes.append(box(x, y+height-weight, x+2*serif+weight, y+height))
    boxes.append(box(x+width-weight, y, x+width, y+weight+1.5*serif))
    return unary_union(boxes)

def make_Ls():
    """Make a pair of block L's."""
    l1 = _single_L(-3.5, -2.25)
    l2 = _single_L(-1.5, -3.5)
    return unary_union([l1, l2])

def sample_spike_train(num_neurons, time_window, rate, seed=0):
    """A random spike train from a Poisson process."""
    npr.seed(seed)
    rate *= np.ones(num_neurons)
    num_spikes = npr.poisson(time_window * rate)
    spikes = [npr.rand(s) * time_window for s in num_spikes]
    spikes = list(map(np.sort, spikes))
    return spikes

def make_spike_train(ax, state, spikes, time_window, bkgd,
                     bottom=-3.5,
                     top=3.5,
                     left=-5.25,
                     right=-1.75,
                     sep_frac=0.2):
    t_start = time_window * (1 - state)
    t_stop = t_start + time_window
    num_neurons = len(spikes)
    spike_height = (top - bottom) / ((1 + sep_frac) * num_neurons)
    spike_sep = sep_frac * spike_height

    lines = []
    for neuron, nspikes in enumerate(spikes):
        # pad the spike train with itself on the left
        nspikes = np.concatenate((nspikes, nspikes + time_window))
        for spike in nspikes:
            if spike < t_start or spike > t_stop:
                continue

            xpos = left + (spike - t_start) * (right - left) / time_window
            ypos = bottom + neuron * (spike_height + spike_sep)

            line = LineString([[xpos, ypos], [xpos, ypos+spike_height]]).buffer(0.01)
            if bkgd.contains(line):
                lines.append(line)

    return lines


def get_code_buffers(file, num_rows=23, buffer_size=100, window_size=45):
    text = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                if line[0] not in ('#', '@', '"'):
                    text.append(line)
    text = ' '.join(text)

    # choose random starting points
    stops = npr.choice(np.arange(buffer_size, len(text)), size=num_rows)
    buffers = [text[stop - buffer_size:stop] for stop in stops]
    return buffers

def make_code_textbox(buffers, state, buffer_size=100, window_size=45):
    start = int((buffer_size - window_size) * (1 - state))

    textbox = '\n'.join([
        buffer[start:start + window_size]
        for buffer in buffers
    ])

    return textbox


def sample_mixture_model(num_points, cache=False):
    if cache:
        fname = "gmm_{}.pkl".format(num_points)
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                results = pickle.load(f)

        else:
            results = _sample_mixture_model(num_points)
            with open(fname, "wb") as f:
                pickle.dump(results, f)
    else:
        results = _sample_mixture_model(num_points)

    return results


def _sample_mixture_model(num_points):
    """Sample a Gaussian mixture model and compute its posterior"""

    means = np.array([
        [2, 3],
        [3, 1.5],
        [4, 0],
        [4.5, -1.5]
    ])
    covariances = np.array([
        [[0.1, 0],
         [0, 0.1]],
        [[0.4, 0.05],
         [0.05, 0.2]],
        [[0.3, -0.05],
         [-0.05, 0.2]],
        [[0.05, 0],
         [0, 0.2]]
        ])

    npr.seed(0)
    data = np.zeros((num_points, 2))
    for n in range(num_points):
        comp = npr.choice(len(means))
        data[n] = npr.multivariate_normal(means[comp], covariances[comp])

    times = npr.rand(num_points)
    perm = np.argsort(times)
    data = data[perm]
    times = times[perm]

    valid = (np.abs(data[:, 1]) < 3.5) & (np.linalg.norm(data, axis=1) < 5.25)
    data = data[valid]
    times = times[valid]
    gmms = []
    return data, times, means, covariances, gmms


def plot_gaussian_2D(mu, lmbda, color='b', num_std=2,
                     centermarker=True, label='',
                     alpha=1., ax=None, artists=None):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2
    ax = ax if ax else plt.gca()

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    point = ax.scatter([mu[0]],[mu[1]],marker='D',color=color,s=4,alpha=alpha, zorder=3) \
            if centermarker else None

    for scale in range(1, num_std+1):
        ax.plot(scale * ellipse[0,:] + mu[0],
                scale * ellipse[1,:] + mu[1],
                linestyle='-', linewidth=2,
                color=color,
                label=label,
                alpha=alpha,
                zorder=3)


def init_kdeplot(bkgd):
    X, Y = np.meshgrid(np.linspace(0, 5, 100),
                       np.linspace(-3.75, 3.75, 100))
    XY = np.column_stack((X.ravel(), Y.ravel()))

    valid = np.array([bkgd.contains(Point(*xy)) for xy in XY])
    return X, Y, XY, valid

def kdeplot(points, X, Y, XY, valid, lengthscale=0.25):
    logpdf = -0.5 * np.sum((XY[:, None, :] - points[None, :, :])**2 / lengthscale**2, axis=-1)
    from scipy.special import logsumexp
    logpdf = logsumexp(logpdf, axis=1)
    logpdf[~valid] = np.nan
    plt.contour(X, Y, np.exp(logpdf).reshape(X.shape), 6, colors=colors["kde"], zorder=3)

def draw_frame(ax, state, spikes, buffers, data, times, means, covariances, gmms, helpers=None):
    ax.cla()
    ax.patch.set_color((0, 0, 0, 0.))

    Ls = make_Ls()
    ax.add_patch(PolygonPatch(Ls, facecolor=colors["Ls"], edgecolor="none", zorder=0))

    # Plot the text box
    textbox = make_code_textbox(buffers, state)
    ax.text(-3.5, 3.4, textbox,
            fontsize=10.,
            fontfamily="monospace",
            verticalalignment='top', color=colors["code"], zorder=1)

    # Overlay background
    bkgd = Point(0, 0).buffer(5.25) - Ls
    ax.add_patch(PolygonPatch(bkgd, facecolor=colors["bkgd"], edgecolor='none', zorder=2))

    # Draw the spike train
    lines = make_spike_train(ax, state, spikes, 10, bkgd)
    for line in lines:
        ax.add_patch(PolygonPatch(line, facecolor='w', edgecolor='w', zorder=3))

    # Plot a contour of the KDE
    if helpers is None:
        helpers = init_kdeplot(bkgd)

    # Plot data points and kernel density estimate
    these_data = data[times < state]
    if len(these_data > 0):
        kdeplot(these_data, *helpers)
    ax.plot(these_data[:, 0], these_data[:, 1], 'wo', markersize=4, zorder=4)

    ax.add_patch(PolygonPatch(bkgd, facecolor='none', edgecolor='k', lw=6, zorder=100))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)

    return helpers

if __name__ == "__main__":

    # Sample random spike train
    spikes = sample_spike_train(20, 10, 0.75)

    # Get code buffers
    buffers = get_code_buffers(TEXTFILE, 25)

    # Sample gaussian mixture model
    data, times, means, covariances, gmms = sample_mixture_model(150)

    # Initialize plot
    fig = plt.figure(figsize=(6, 6))
    # fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    # ax.patch.set_color("gray")
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)

    # # Draw a single frame
    # draw_frame(ax, 0, spikes, buffers, data, times, means, covariances, gmms)
    # plt.show()
    # plt.savefig("logo.png", dpi=300)
    # plt.savefig("logo.pdf")

    # Save still frames and convert to gif with
    #   convert -delay 3 -loop 0 _stills/frame_{0..99}.png logo.gif
    # The `convert` util comes with ImageMagick
    thetas = np.linspace(0, 1, 100)
    helpers = None
    for i in trange(len(thetas)):
        helpers = draw_frame(ax, thetas[i], spikes, buffers, data, times, means, covariances, gmms, helpers=helpers)
        plt.savefig("_stills_code/frame_{}.png".format(i), dpi=50)
