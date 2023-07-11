""" 
lab_utils_uni.py
    routines used in Course 1, Week2, labs1-3 dealing with single variables (univariate)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from ipywidgets import interact
from .common import compute_cost
from .common import dlblue, dlorange, dldarkred, dlmagenta, dlpurple, dlcolors

plt.style.use("./deeplearning.mplstyle")
n_bin = 5
dlcm = LinearSegmentedColormap.from_list("dl_map", dlcolors, N=n_bin)

##########################################################
# Plotting Routines
##########################################################


def plt_x_y(X, y, pred=None, ax=None, title=None, xlabel=None, ylabel=None):
    """Plot X and y, and optionally pred"""
    if not ax:
        fig, ax = plt.subplots(1, 1)

    ax.scatter(X, y, marker="x", c="r", label="Actual Value")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if pred is not None:
        ax.plot(X, pred, c=dlblue, label="Our Prediction")

    ax.legend()

def soup_bowl():
    """Create figure and plot with a 3D projection"""
    fig = plt.figure(figsize=(8, 8))

    # Plot configuration
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)

    # Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    # Get the z value for a bowl-shaped cost function
    z = np.zeros((len(w), len(b)))
    j = 0
    for x in w:
        i = 0
        for y in b:
            z[i, j] = x**2 + y**2
            i += 1
        j += 1

    # Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w, b)

    # Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color="k", alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$\n [You can rotate this figure]", size=15)

    plt.show()


def inbounds(a, b, xlim, ylim):
    xlow, xhigh = xlim
    ylow, yhigh = ylim
    ax, ay = a
    bx, by = b
    if (
        (ax > xlow and ax < xhigh)
        and (bx > xlow and bx < xhigh)
        and (ay > ylow and ay < yhigh)
        and (by > ylow and by < yhigh)
    ):
        return True
    return False


def plt_contour_wgrad(
    x,
    y,
    hist,
    f_cost,
    ax=None,
    w_range=[-100, 500, 5],
    b_range=[-500, 500, 5],
    contours=[0.1, 50, 1000, 5000, 10000, 25000, 50000],
    resolution=5,
    w_final=200,
    b_final=100,
    step=10,
):
    """Plot contour plot of cost function with gradient descent path
    
    @param x: x values
    @param y: y values
    @param hist: history of gradient descent (Cost function)
    @param ax: axis to plot on
    @param w_range: range of w values
    @param b_range: range of b values
    @param contours: contour lines to plot
    @param resolution: resolution of contour plot
    @param w_final: final w value
    @param b_final: final b value
    @param step: step size of gradient descent

    @return: None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 16))

    b0, w0 = np.meshgrid(np.arange(*b_range), np.arange(*w_range))
    z = np.zeros_like(b0)
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            z[i][j] = f_cost(x, y, w0[i][j], b0[i][j])

    CS = ax.contour(
        w0,
        b0,
        z,
        contours,
        linewidths=2,
        colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple],
    )
    ax.clabel(CS, inline=1, fmt="%1.0f", fontsize=10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("Contour plot of cost J(w,b), vs b,w with path of gradient descent")
    w = w_final
    b = b_final
    ax.hlines(b, ax.get_xlim()[0], w, lw=2, color=dlpurple, ls="dotted")
    ax.vlines(w, ax.get_ylim()[0], b, lw=2, color=dlpurple, ls="dotted")

    base = hist[0]
    for point in hist[0::step]:
        edist = np.sqrt((base[0] - point[0]) ** 2 + (base[1] - point[1]) ** 2)
        if edist > resolution or point == hist[-1]:
            if inbounds(point, base, ax.get_xlim(), ax.get_ylim()):
                plt.annotate(
                    "",
                    xy=point,
                    xytext=base,
                    xycoords="data",
                    arrowprops={"arrowstyle": "->", "color": "r", "lw": 3},
                    va="center",
                    ha="center",
                )
            base = point
    return

# draw derivative line
# y = m*(x - x1) + y1
def add_line(dj_dx, x1, y1, d, ax):
    x = np.linspace(x1 - d, x1 + d, 50)
    y = dj_dx * (x - x1) + y1
    ax.scatter(x1, y1, color=dlblue, s=50)
    ax.plot(x, y, "--", c=dldarkred, zorder=10, linewidth=1)
    xoff = 30 if x1 == 200 else 10
    ax.annotate(
        r"$\frac{\partial J}{\partial w}$ =%d" % dj_dx,
        fontsize=14,
        xy=(x1, y1),
        xycoords="data",
        xytext=(xoff, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment="left",
        verticalalignment="top",
    )


def plt_gradients(x_train, y_train, f_compute_cost, f_compute_gradient):
    # ===============
    #  First subplot
    # ===============
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Print w vs cost to see minimum
    fix_b = 100
    w_array = np.linspace(-100, 500, 50)
    w_array = np.linspace(0, 400, 50)
    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = f_compute_cost(x_train, y_train, tmp_w, fix_b)
    ax[0].plot(w_array, cost, linewidth=1)
    ax[0].set_title("Cost vs w, with gradient; b set to 100")
    ax[0].set_ylabel("Cost")
    ax[0].set_xlabel("w")

    # plot lines for fixed b=100
    for tmp_w in [100, 200, 300]:
        fix_b = 100
        dj_dw, dj_db = f_compute_gradient(x_train, y_train, tmp_w, fix_b)
        j = f_compute_cost(x_train, y_train, tmp_w, fix_b)
        add_line(dj_dw, tmp_w, j, 30, ax[0])

    # ===============
    # Second Subplot
    # ===============

    tmp_b, tmp_w = np.meshgrid(np.linspace(-200, 200, 10), np.linspace(-100, 600, 10))
    U = np.zeros_like(tmp_w)
    V = np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            U[i][j], V[i][j] = f_compute_gradient(
                x_train, y_train, tmp_w[i][j], tmp_b[i][j]
            )
    X = tmp_w
    Y = tmp_b
    n = -2
    color_array = np.sqrt(((V - n) / 2) ** 2 + ((U - n) / 2) ** 2)

    ax[1].set_title("Gradient shown in quiver plot")
    Q = ax[1].quiver(
        X,
        Y,
        U,
        V,
        color_array,
        units="width",
    )
    ax[1].quiverkey(
        Q, 0.9, 0.9, 2, r"$2 \frac{m}{s}$", labelpos="E", coordinates="figure"
    )
    ax[1].set_xlabel("w")
    ax[1].set_ylabel("b")
