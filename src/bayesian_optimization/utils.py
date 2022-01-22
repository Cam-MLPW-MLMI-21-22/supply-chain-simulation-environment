import numpy as np
from matplotlib import pyplot as plt
from pylab import *


def plot_3d_boundary(X, Y, mesh_X, mesh_Y,
                     mu_plot, var_plot,
                     elev,
                     angle,
                     z_lims=None,
                     plot_new=False,
                     new_X=None,
                     new_Y=None,
                     dpi=100,
                     plot_ci=True,
                     title=None,
                     save_fig_path=None,
                     fig=None,
                     ax=None,):

    if not fig or not ax:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.gca(projection='3d')

    surf = ax.plot_surface(mesh_X, mesh_Y, mu_plot.reshape(
        mesh_X.shape), cmap='viridis', linewidth=0, antialiased=False, alpha=0.60)

    if plot_ci:
        surf_var = ax.plot_surface(mesh_X, mesh_Y, (mu_plot-var_plot).reshape(
            (1000, 21)), cmap='viridis', linewidth=0, antialiased=False, alpha=0.20)
        surf_var = ax.plot_surface(mesh_X, mesh_Y, (mu_plot+var_plot).reshape(
            (1000, 21)), cmap='viridis', linewidth=0, antialiased=False, alpha=0.20)

    ax.scatter(X[:, 0].flatten(), X[:, 1].flatten(), Y.flatten(),
               s=100, marker="o", color="b", label="Initial observations")

    if plot_new:
        ax.scatter(new_X[:, 0].flatten(), new_X[:, 1].flatten(),
                   new_Y.flatten(), marker="x", color="r", label="All observations", s=100)

    ax.grid(True)
    ax.set_xlabel("Num batteries")
    ax.set_ylabel("Battery capacity")
    ax.set_zlabel("Cumulative reward")

    if z_lims:
        ax.set_zlim(z_lims)

    if title:
        plt.title(title)
    plt.legend(loc='upper right', prop={'size': 15})

    if save_fig_path:
        for theta in range(0, 360, 10):
            ax.view_init(elev=elev, azim=theta)
            plt.savefig(save_fig_path+"_{}.png".format(theta))

    ax.view_init(elev=elev, azim=angle)
    return fig, ax


def plot_2d_boundary(X, Y, mesh_X, mesh_Y,
                     mu_plot, var_plot,
                     plot_new=False,
                     new_X=None,
                     new_Y=None,
                     dpi=100,
                     plot_ci=True,
                     title=None,
                     save_fig_path=None,
                     fig=None,
                     ax=None,
                     plot_func='plot_surface',
                     plot_kwargs={'linewidth': 0, 'antialiased': False}):

    if not fig or not ax:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.gca(projection='3d')

    surf = getattr(ax, plot_func)(mesh_X, mesh_Y, mu_plot.reshape(
        mesh_X.shape), cmap='viridis', alpha=0.50, **plot_kwargs)

    if plot_ci:
        surf_var = getattr(ax, plot_func)(mesh_X, mesh_Y, (mu_plot-var_plot).reshape(
            (1000, 21)), cmap='viridis', alpha=0.20, **plot_kwargs)
        surf_var = getattr(ax, plot_func)(mesh_X, mesh_Y, (mu_plot+var_plot).reshape(
            (1000, 21)), cmap='viridis', alpha=0.20, **plot_kwargs)

    ax.scatter(X[:, 0].flatten(), X[:, 1].flatten(), s=100,
               marker="o", color="b", label="Initial observations")

    if plot_new:
        ax.scatter(new_X[:, 0].flatten(), new_X[:, 1].flatten(),
                   s=100, marker="x", color="r", label="All observations")

    ax.grid(True)
    ax.set_xlabel("Num batteries")
    ax.set_ylabel("Battery capacity")

    if title:
        plt.title(title)
    plt.legend(loc='upper right', prop={'size': 15})

    if save_fig_path:
        for theta in range(0, 360, 10):
            plt.savefig(save_fig_path+"_{}.png".format(theta))

    return fig, ax

def plot_3d_observed_rewards(X, Y,
                             elev,
                             angle,
                             z_lims=None,
                             plot_new=None,
                             new_X=None,
                             new_Y=None,
                             title=None,
                             dpi=100,
                             save_fig_path=None):

    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = plt.axes(projection='3d')

    ax.scatter(X[:, 0].flatten(), X[:, 1].flatten(), Y.flatten(),
               s=100, marker="o", color="b", label="Initial observations")

    if plot_new:
        im = ax.plot_trisurf(new_X[:, 0].flatten(), new_X[:, 1].flatten(
        ), new_Y.flatten(), cmap='viridis', alpha=0.70)

        ax.scatter(new_X[:, 0].flatten(), new_X[:, 1].flatten(),
                   new_Y.flatten(), marker="x", color="r", label="All observations", s=100)

        min_X = new_X[np.argmin(new_Y)]
        min_Y = np.min(new_Y)
        ax.scatter(min_X[0], min_X[1], min_Y, c='black',
                   marker='D', label="Minimum", s=200)
    else:
        im = ax.plot_trisurf(X[:, 0].flatten(), X[:, 1].flatten(
        ), Y.flatten(), cmap='viridis', alpha=0.70)

    ax.legend(loc=1, prop={'size': 15})
    ax.set_xlabel("Num batteries")
    ax.set_ylabel("Battery capacity")
    ax.set_zlabel("Cumulative reward")

    ax.view_init(elev=elev, azim=angle)

    if z_lims:
        ax.set_zlim(z_lims)

    ax.grid(True)
    if title == None:
        plt.title("Contour of observed rewards")
    else:
        plt.title(title)

    if save_fig_path:
        for theta in range(0, 360, 10):
            ax.view_init(elev=elev, azim=theta)
            plt.savefig(save_fig_path+"_{}.png".format(theta))

    return fig, ax
