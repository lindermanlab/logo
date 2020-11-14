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
    gmm="#F9F6EF"
)


def gear(center,
         inner_radius,
         outer_radius,
         num_teeth,
         taper=.25,
         rotation=0,
         num_pts_per_tooth=10,
         hollow_radius=None):
    """Draw a gear with specified size and number of teeth."""
    tooth_rads = 2 * np.pi / (2 * num_teeth)

    # Make the list of points in polar coordinates
    angles = []
    radii = []
    offset = 0
    for tooth in range(num_teeth):
        # add the tooth with a taper
        radii.extend(outer_radius * np.ones(num_pts_per_tooth))
        angles.extend(np.linspace(offset + taper * tooth_rads,
                                  offset + (1 - taper) * tooth_rads,
                                  num_pts_per_tooth))
        offset += tooth_rads

        # add the gap
        radii.extend(inner_radius * np.ones(num_pts_per_tooth))
        angles.extend(np.linspace(offset,
                                  offset + tooth_rads,
                                  num_pts_per_tooth))
        offset += tooth_rads

    # Close the path
    radii.extend([outer_radius])
    angles.extend([taper * tooth_rads])

    # Convert to xy coordinates
    xys = np.column_stack([
        np.array(radii) * np.cos(angles),
        np.array(radii) * np.sin(angles)])

    # Counter rotate by half a tooth so that it starts at a
    # more natural position
    theta = rotation - tooth_rads / 2

    # Rotate and center
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    shell = xys @ R.T + np.array(center)

    # Cut out the center
    holes = []
    if hollow_radius is not None:
        assert hollow_radius < inner_radius
        # The inner circle has to be specified in the opposite direction.
        # Not really sure why...
        holes.append(np.column_stack([
            center[0] + hollow_radius * np.cos(np.linspace(2*np.pi, 0, 100)),
            center[1] + hollow_radius * np.sin(np.linspace(2*np.pi, 0, 100)),
            ]))

    return Polygon(shell, holes)


def make_wheel(center,
               inner_radius,
               outer_radius,
               hub_radius,
               spoke_radius,
               rotation=0,
               num_spokes=6):
    """Draw a wheel with desired specs"""
    tire = Point(0, 0).buffer(outer_radius) \
         - Point(0, 0).buffer(inner_radius)

    hub = Point(0, 0).buffer(hub_radius)

    spokes = []
    for th in np.linspace(0, 2*np.pi, num_spokes, endpoint=False):
        ls = LineString(
            [(0, 0),
            inner_radius * np.array([np.cos(th), np.sin(th)])])
        spokes.append(ls.buffer(spoke_radius))

    wheel = unary_union([tire, hub] + spokes)
    return translate(rotate(wheel, rotation, use_radians=True), *center)


def make_piston(height, endpoint):
    """Make a fancy piston that rotates on an axis"""

    rod = LineString([(0, height), endpoint]).buffer(0.08)
    angle = np.arctan2(endpoint[0], height - endpoint[1])
    end = LineString([(0, height), (0, height) + 0.5 * np.array([np.sin(angle), -np.cos(angle)])]).buffer(.16)
    rod = unary_union([rod, end])
    axis = Point(0, height).buffer(0.08)
    return rod - axis

def make_steam_engine(sun_center, state,
                      piston_length=2.5,
                      chamber_width=2.5,
                      chamber_top=4,
                      chamber_left=-1.25,
                      chamber_right=3,
                      gear_inner_radius=0.66,
                      gear_outer_radius=0.8,
                      gear_hollow_radius=0.5,
                      gear_num_teeth=12,
                      wheel_inner_radius=2.75,
                      wheel_outer_radius=3.,
                      wheel_hub_radius=0.25,
                      wheel_spoke_radius=0.025,
                      wheel_num_spokes=12,
                      wheel2_center=(-2, 4.5),
                      wheel2_inner_radius=0.5,
                      wheel2_outer_radius=0.6,
                      wheel2_hub_radius=0.05,
                      wheel2_num_spokes=8,
                      wheel2_spoke_radius=0.01,):
    """Make a steam engine schematic from a sun and planet gearing.
    (https://en.wikipedia.org/wiki/Sun_and_planet_gear#/media/File:Sun_and_planet_gears.gif
    and https://en.wikipedia.org/wiki/Steam_engine#/media/File:Steam_engine_in_action.gif)
    It drives two wheels, the flywheel drives wheel2 via a belt."""
    assert piston_length > 2.1 * gear_outer_radius

    # Planet goes at state (in radians) angle from center
    planet_center = np.array([
        1.02 * (gear_inner_radius + gear_outer_radius) * np.cos(state),
        1.02 * (gear_inner_radius + gear_outer_radius) * np.sin(state)])

    # Determine the piston location from the planet center
    # Assuming piston is vertically oriented, the piston lies
    # on the y axis and the hypotenuse of the right triangle
    # formed beyween the y axis and the planet is known length.
    piston_height = np.sqrt(piston_length**2 - planet_center[0]**2) + planet_center[1]
    assert np.isreal(piston_height)

    # Planet is rotated so that it always faces the piston
    rotation = np.arctan2(piston_height - planet_center[1],
                          0 - planet_center[0])

    planet = gear(planet_center,
                  gear_inner_radius,
                  gear_outer_radius,
                  gear_num_teeth,
                  hollow_radius=gear_hollow_radius,
                  rotation=rotation)

    # Draw the piston and the chamber above
    piston = make_piston(piston_height, planet_center)
    if piston_height+.15 > 2.7:
        pad = box(chamber_left, piston_height-0.15, chamber_right, piston_height+0.15)
        chamber = box(-1.25, piston_height+0.15, 1.25, chamber_top)
    else:
        pad = box(-.5, piston_height-0.15, .5, piston_height+0.15)
        chamber = unary_union(
            [box(-1.25, 3, 1.25, chamber_top),
             box(-.5, piston_height+0.15, .5, chamber_top)]
        )


    # The sun goes at the center. It has to be rotated to
    # be compatible with the planet.  Do this by calculating the
    # rotation of the planet when it is horizontal to the sun
    # (i.e. when state=0), then rotate the sun by half a tooth
    sun_offset = np.arctan2(piston_length,
                            1.02 * (gear_outer_radius + gear_inner_radius))
    sun_offset += np.pi / gear_num_teeth
    sun_rotation = sun_offset + state
    sun = gear((0, 0),
               gear_inner_radius,
               gear_outer_radius,
               gear_num_teeth,
               hollow_radius=gear_hollow_radius,
               rotation=sun_rotation)

    # Draw a wheel centered from the sun
    wheel1 = make_wheel((0, 0),
                       wheel_inner_radius,
                       wheel_outer_radius,
                       wheel_hub_radius,
                       num_spokes=wheel_num_spokes,
                       spoke_radius=wheel_spoke_radius,
                       rotation=sun_rotation)

    # Draw a second wheel that is driven by the first
    wheel2 = make_wheel(wheel2_center,
                        inner_radius=wheel2_inner_radius,
                        outer_radius=wheel2_outer_radius,
                        hub_radius=wheel2_hub_radius,
                        num_spokes=wheel2_num_spokes,
                        spoke_radius=wheel2_spoke_radius,
                        rotation=sun_rotation * wheel_outer_radius/wheel2_outer_radius)

    # Make a belt
    belt = unary_union([wheel1, wheel2]).convex_hull.boundary.buffer(0.02)
    # belt = scale(belt, xfact=1.015, yfact=1.015) - belt

    # translate each part
    wheel1, wheel2, belt, sun, planet, piston, pad, chamber = \
        map(partial(translate, xoff=sun_center[0], yoff=sun_center[1]),
            [wheel1, wheel2, belt, sun, planet, piston, pad, chamber])

    # Make patches
    patches = []
    for part, name in zip(
        [wheel1, wheel2, belt, sun, planet, piston, pad, chamber],
        ["wheel1", "wheel2", "belt", "sun", "planet", "piston", "pad", "chamber"]):
        patch = PolygonPatch(part, facecolor=colors[name],
                             alpha=0.8,  edgecolor='k')

        if name == "chamber":
            patch.set_hatch('.')
            patch.set_color("#0098db")
            patch.set_alpha(0.25)

        patches.append(patch)

    return patches

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
    t_start = time_window * state / (2 * np.pi)
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
            # if line.within(bkgd):
            if bkgd.contains(line):
                lines.append(line)

    return lines


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

    # Precompute Gaussian mixture solutions
    # print("Fitting GMMs")
    # from sklearn.mixture import GaussianMixture
    # from sklearn.model_selection import cross_validate
    gmms = []
    # for n in range(len(data)):
    #     if n < 6:
    #         gmms.append(None)
    #         continue

    #     scores = []
    #     models = []
    #     for k in range(1, 6):
    #         models.append(GaussianMixture(k).fit(data[:n+1]))
    #         scores.append(cross_validate(models[-1], data[:n+1])["test_score"].mean())
    #     best = np.argmax(scores)
    #     gmms.append(models[best])
    # print("Done")

    return data, times, means, covariances, gmms


def plot_gaussian_2D(mu, lmbda, color='b', num_std=2,
                     centermarker=True, label='',
                     alpha=1., ax=None, artists=None):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2
    ax = ax if ax else plt.gca()

    # TODO if update alpha=0. and our previous alpha is 0., we don't need to
    # dirty the artist

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    point = ax.scatter([mu[0]],[mu[1]],marker='D',color=color,s=4,alpha=alpha) \
            if centermarker else None

    for scale in range(1, num_std+1):
        ax.plot(scale * ellipse[0,:] + mu[0],
                scale * ellipse[1,:] + mu[1],
                linestyle='-', linewidth=2,
                color=color,
                label=label,
                alpha=alpha)


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
    plt.contour(X, Y, np.exp(logpdf).reshape(X.shape), 6, colors='gray')

def draw_frame(ax, state, spikes, data, times, means, covariances, gmms, helpers=None):
    ax.cla()
    ax.patch.set_color((0, 0, 0, 0.))
    center = (-.25, -1.75)

    # patches = make_steam_engine(center, state)
    # for patch in patches:
    #     ax.add_patch(patch)

    Ls = make_Ls()
    ax.add_patch(PolygonPatch(Ls, facecolor=colors["Ls"], edgecolor="none"))
    bkgd = Point(0, 0).buffer(5.25) - Ls
    ax.add_patch(PolygonPatch(bkgd, facecolor=colors["bkgd"], edgecolor='none'))

    # Draw the spike train
    lines = make_spike_train(ax, state, spikes, 10, bkgd)
    for line in lines:
        ax.add_patch(PolygonPatch(line, facecolor='w', edgecolor='w'))

    # Plot a contour of the KDE
    if helpers is None:
        helpers = init_kdeplot(bkgd)

    # if len(these_data > 0):
    #     kdeplot(these_data, *helpers)
    for mean, covariance in zip(means, covariances):
        plot_gaussian_2D(mean, covariance, color=colors["gmm"])

    # Plot data points
    these_data = data[times < 1 - state / (2*np.pi)]
    ax.plot(these_data[:, 0], these_data[:, 1], 'wo', markersize=4)

    ax.add_patch(PolygonPatch(bkgd, facecolor='none', edgecolor='k', lw=6, zorder=100))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)

    return helpers

if __name__ == "__main__":

    # Sample random spike train
    spikes = sample_spike_train(20, 10, 0.75)

    # Sample gaussian mixture model
    data, times, means, covariances, gmms = sample_mixture_model(150)

    # Initialize plot
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    # ax.patch.set_color("gray")
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)

    # # Draw a single frame
    draw_frame(ax, 1 /4 * np.pi, spikes, data, times, means, covariances, gmms)
    plt.savefig("logo.png", dpi=50)
    plt.savefig("logo.pdf")
    plt.show()

    # Save still frames and convert to gif with
    #   convert -delay 3 -loop 0 _stills/frame_{0..99}.png logo.gif
    # The `convert` util comes with ImageMagick
    # thetas = np.linspace(0, 2*np.pi, 100)[::-1]
    # helpers = None
    # for i in trange(len(thetas)):
    #     helpers = draw_frame(ax, thetas[i], spikes, data, times, gmms, helpers=helpers)
    #     plt.savefig("_stills/frame_{}.png".format(i))
