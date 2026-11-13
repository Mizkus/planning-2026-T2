from pathlib import Path
from queue import PriorityQueue

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d

import utils


def dijkstra(environment, start, goal, heuristic=0):
    current = start
    cost = 0
    q = PriorityQueue()
    q.put((0, 0, start))
    curr_env = environment.copy()
    dist = {start: 0}
    prev = {}
    visited = 0

    def add_neighbours(point, cost):
        x, y = point
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dx, dy = x + i, y + j
            if curr_env[dx, dy] == 0:
                t = cost + 1
                if t < dist.get((dx, dy), float('inf')):
                    dist[(dx, dy)] = t
                    prev[(dx, dy)] = point
                    h = abs(dx - goal[0]) + abs(dy - goal[1]) if heuristic else 0
                    q.put((t + h, t, (dx, dy)))

    while not q.empty():
        _, cost, current = q.get()
        if curr_env[current] == 1:
            continue
        visited += 1
        if current == goal:
            break
        curr_env[current] = 1
        add_neighbours(current, cost)

    if current != goal:
        return [], -1, visited

    plan = [goal]
    while plan[-1] != start:
        plan.append(prev[plan[-1]])
    plan.reverse()
    return plan, cost, visited


def dijkstra_rod(environment, start, goal, heuristic=0):
    current = start
    cost = 0
    q = PriorityQueue()
    q.put((0, 0, start))
    curr_env = environment.copy()
    dist = {start: 0}
    prev = {}
    visited = 0

    def add_neighbours(point, cost):
        x, y, z = point
        for i, j, k in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)]:
            dx, dy, dz = x + i, y + j, (z + k) % 4
            if curr_env[dx, dy, dz] == 0:
                t = cost + 1
                if t < dist.get((dx, dy, dz), float('inf')):
                    dist[(dx, dy, dz)] = t
                    prev[(dx, dy, dz)] = point
                    if heuristic == 1:
                        h = abs(dx - goal[0]) + abs(dy - goal[1]) 
                    elif heuristic == 2:
                        h = abs(dx - goal[0]) + abs(dy - goal[1]) + min(abs(dz - goal[2]), 4 - abs(dz - goal[2]))
                    else:
                        h = 0
                    q.put((t + h, t, (dx, dy, dz)))

    while not q.empty():
        _, cost, current = q.get()
        if curr_env[current] == 1:
            continue
        visited += 1
        if current == goal:
            break
        curr_env[current] = 1
        add_neighbours(current, cost)

    if current != goal:
        return [], -1, visited

    plan = [goal]
    while plan[-1] != start:
        plan.append(prev[plan[-1]])
    plan.reverse()
    return plan, cost, visited


def save_figure(path):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")


def main():
    root = Path(__file__).resolve().parent
    output = root / "results"
    output.mkdir(exist_ok=True)

    with np.load(root / "data_ps1.npz") as data:
        environment = data["environment"]
        rod = data["rod"]

    start = (6, 6)
    goal = (85, 85)
    point_runs = [
        ("Dijkstra", 0, "point_solve.gif"),
        ("A* h=0", 0, "point_solve_a_star_zero.gif"),
        ("A* L1", 1, "point_solve_a_star.gif"),
    ]
    rows = []
    for name, heuristic, filename in point_runs:
        plan, cost, visited = dijkstra(environment, start, goal, heuristic)
        if not plan:
            raise RuntimeError(f"{name}: point path not found")
        rows.append({"Algorithm": name, "Visited": visited, "Cost": cost})
        np.save(output / filename.replace(".gif", "_plan.npy"), np.asarray(plan))
        utils.plotting_results_2d(environment, plan, save_path=str(root / filename))
        plt.close("all")
        fig, ax = utils.plotting_results_2d_arrows(environment, plan)
        ax.set_title(name)
        save_figure(output / filename.replace(".gif", ".png"))
    point_results = pd.DataFrame(rows)
    point_results.to_csv(output / "task1_results.csv", index=False)
    print("Task 1")
    print(point_results.to_string(index=False))

    plt.figure(figsize=(8, 8))
    plt.imshow(environment, cmap="viridis", vmin=0, vmax=1)
    save_figure(output / "workspace.png")

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for i in range(4):
        axes[i].imshow(rod[:, :, i], cmap="viridis", vmin=0, vmax=1)
        axes[i].set_title(f"{i * 45}°")
    save_figure(output / "rod_orientations.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(utils.plot_environment_3d(environment, rod, (20, 40, 1)))
    save_figure(output / "rod_on_map.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    envs = []
    for i in range(4):
        conv = convolve2d(environment, rod[:, :, i], boundary="symm", mode="same")
        norm_conv = utils.normalize_image(conv)
        envs.append(norm_conv)
        axes[i // 2, i % 2].imshow(norm_conv, cmap="viridis", vmin=0, vmax=1)
        axes[i // 2, i % 2].set_title(f"{i * 45}°")
    envs = np.array(envs).transpose(1, 2, 0)
    save_figure(output / "configuration_space.png")
    np.save(output / "configuration_space.npy", envs)
    print(f"Task 2: C-space shape = {envs.shape}")

    start = (6, 6, 2)
    goal = (85, 85, 0)
    rod_runs = [
        ("Dijkstra", 0, "rod_solve.gif"),
        ("A* L1", 1, "rod_solve_a_star.gif"),
        ("A* L1 + angle", 2, "rod_solve_a_star_ang.gif"),
    ]
    rows = []
    for name, heuristic, filename in rod_runs:
        plan, cost, visited = dijkstra_rod(envs, start, goal, heuristic)
        if not plan:
            raise RuntimeError(f"{name}: rod path not found")
        rows.append({"Algorithm": name, "Visited": visited, "Cost": cost})
        np.save(output / filename.replace(".gif", "_plan.npy"), np.asarray(plan))
        utils.plotting_results_3d(environment, rod, plan, save_path=str(root / filename))
        plt.close("all")
        fig, ax = utils.plotting_results_3d_arrows(environment, plan)
        save_figure(output / filename.replace(".gif", ".png"))
    rod_results = pd.DataFrame(rows)
    rod_results.to_csv(output / "task3_results.csv", index=False)
    print("Task 3")
    print(rod_results.to_string(index=False))


if __name__ == "__main__":
    main()

