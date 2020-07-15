import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Initialize plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes((0, 0, 1, 1))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Draw a stream plot in the background
def stream():
    w = 5.25
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2
    speed = np.sqrt(U**2 + V**2)

    bad = np.where(np.sqrt(X**2 + Y**2) > 5.25)
    U[bad] = np.nan
    V[bad] = np.nan
    ax.streamplot(X, Y, U, V, density=[3, 3], color='k', zorder=-10)

# Draw a block L
def block_L(xy, height=5.75, width=5, weight=1, serif=.75):
    # (x, y) sets the lower left corner of the L
    x, y = xy
    # Create the vertical and horizontal bar
    r1 = Rectangle((x+serif, y), weight, height)
    r2 = Rectangle((x, y), width, weight)
    # Add the serifs
    r3 = Rectangle((x, y+height-weight), 2*serif+weight, weight)
    r4 = Rectangle((x+width-weight, y), weight, 1.5 * serif+weight)
    for r in [r1, r2, r3, r4]:
        ax.add_patch(r)

# stream()
block_L((-3.5, -2.25))
block_L((-1.5, -3.5))

# Encapsulate it in a circle
circle = Circle((0,0), radius=5.25, facecolor="none", edgecolor="k", linewidth=4)
ax.add_patch(circle)
    
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)

plt.show()

