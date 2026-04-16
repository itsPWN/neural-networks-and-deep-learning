"""valley2.py
~~~~~~~~~~~~~

Plots a function of two variables to minimize.  The function is a
fairly generic valley function.

Note that this is a duplicate of valley.py, but omits labels on the
axis.  It's bad practice to duplicate in this way, but I had
considerable trouble getting matplotlib to update a graph in the way I
needed (adding or removing labels), so finally fell back on this as a
kludge solution.

"""

from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X = numpy.arange(-1, 1, 0.1)
Y = numpy.arange(-1, 1, 0.1)
X, Y = numpy.meshgrid(X, Y)
Z = X**2 + Y**2

colortuple = ("w", "b")
colors = numpy.empty(X.shape, dtype=str)
for x in range(len(X)):
    for y in range(len(Y)):
        colors[x, y] = colortuple[(x + y) % 2]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0)

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 2)
ax.w_xaxis.set_major_locator(LinearLocator(3))
ax.w_yaxis.set_major_locator(LinearLocator(3))
ax.w_zaxis.set_major_locator(LinearLocator(3))
ax.text(1.79, 0, 1.62, "$C$", fontsize=20)

plt.show()
