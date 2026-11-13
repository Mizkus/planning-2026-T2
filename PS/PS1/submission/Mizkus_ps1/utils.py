
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Patch


def normalize_image(img: np.ndarray, threshold: float = 0.1):
    dims = img.shape
    env = np.ones(dims)
    z = np.where(img < threshold)
    env[z] = 0.0
    return env


def plot_environment_3d(img: np.ndarray, obj: np.ndarray, state: tuple):
    dims = obj.shape
    dim_x = int((dims[0] - 1) / 2)
    dim_y = int((dims[1] - 1) / 2)
    merged_img = np.copy(img)
    merged_img[state[0] - dim_x:state[0] + dim_x + 1, state[1] - dim_y:state[1] + dim_y + 1] += obj[:, :, state[2]] * 0.5
    return merged_img


def plot_environment_2d(img: np.ndarray, state: tuple):
    merged_img = np.copy(img)
    merged_img[state[0], state[1]] += 0.5 
    return merged_img

def plotting_results_3d(environment: np.ndarray, rod: np.ndarray, plan: list, save_path: str = 'rod_solve.gif'):

    fig = plt.figure()
    imgs = []

    for s in plan:
        im = plot_environment_3d(environment, rod, s)
        plot = plt.imshow(im)
        imgs.append([plot])

    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)

    ani.save(save_path)

    plt.show()

def plotting_results_2d(environment: np.ndarray, plan: list, save_path: str = 'point_solve.gif'):

    fig = plt.figure()
    imgs = []

    for s in plan:
        im = plot_environment_2d(environment, s)
        plot = plt.imshow(im)
        imgs.append([plot])

    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)

    ani.save(save_path)

    plt.show()



def plotting_results_2d_arrows(environment: np.ndarray, plan: list, save_path: str = None):
    plan = np.asarray(plan, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap('viridis')
    ax.imshow(environment, cmap=cmap, vmin=0, vmax=1, origin='upper', interpolation='none')

    for x, y in plan:
        ax.add_patch(Rectangle(
            (y - 0.5, x - 0.5),
            1,
            1,
            facecolor=cmap(0.5),
            edgecolor='none',
            zorder=2,
        ))

    ax.add_patch(Rectangle(
        (plan[0, 1] - 0.5, plan[0, 0] - 0.5),
        1,
        1,
        facecolor='green',
        edgecolor='black',
        zorder=3,
    ))
    ax.add_patch(Rectangle(
        (plan[-1, 1] - 0.5, plan[-1, 0] - 0.5),
        1,
        1,
        facecolor='red',
        edgecolor='black',
        zorder=3,
    ))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, environment.shape[1] - 0.5)
    ax.set_ylim(environment.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, environment.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, environment.shape[0], 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.25)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xlabel('y')
    ax.set_ylabel('x')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()
    return fig, ax


def plotting_results_3d_arrows(environment: np.ndarray, plan: list, save_path: str = None):
    plan = np.asarray(plan, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap('viridis')
    colors = ['#E63946', '#FFB703', '#3A86FF', '#B5179E']
    ax.imshow(environment, cmap=cmap, vmin=0, vmax=1, origin='upper', interpolation='none')

    for x, y, z in plan:
        ax.add_patch(Rectangle(
            (y - 0.5, x - 0.5),
            1,
            1,
            facecolor=colors[z],
            edgecolor='none',
            zorder=2,
        ))

    ax.add_patch(Rectangle(
        (plan[0, 1] - 0.5, plan[0, 0] - 0.5),
        1,
        1,
        facecolor='green',
        edgecolor='black',
        zorder=3,
    ))
    ax.add_patch(Rectangle(
        (plan[-1, 1] - 0.5, plan[-1, 0] - 0.5),
        1,
        1,
        facecolor='red',
        edgecolor='black',
        zorder=3,
    ))

    ax.legend(handles=[Patch(color=colors[i], label=f'{i * 45}°') for i in range(4)] +
              [Patch(color='green', label='Start'), Patch(color='red', label='Goal')],
              loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=6, frameon=False)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, environment.shape[1] - 0.5)
    ax.set_ylim(environment.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, environment.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, environment.shape[0], 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.25)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xlabel('y')
    ax.set_ylabel('x')

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()
    return fig, ax

