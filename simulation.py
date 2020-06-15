import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def vector_product(x, y):
    return x[:, :, :, 0] * y[:, :, :, 1] - x[:, :, :, 1] * y[:, :, :, 0]


def intersect(x1, d1, x2, d2):
    """
    Arguments:
    x1, d1: (n, d, 2)
    x2, d2: (m, 2)
    return: tu: (n, m, 2)
        x1[k] + tu[k, l, 0] * d1[k] == x2[l] + tu[k, l, 1] * d2[l]
    """
    x1 = x1[:, :, None, :]
    x2 = x2[None, None, :, :]
    d1 = d1[:, :, None, :]
    d2 = d2[None, None, :, :]
    rs = vector_product(d1, d2)
    t = vector_product(x2 - x1, d2) / rs
    u = vector_product(x2 - x1, d1) / rs
    tu = np.stack([t, u], 3)
    m = np.any((tu < 0) | (tu > 1), axis=3)
    tu[m] = np.nan
    return tu


def make_rectangle(x1, x2):
    """
    Arguments:
    x1: lower left corner
    x2: upper right corner
    """
    dx = np.array([x2[0] - x1[0], 0])
    dy = np.array([0, x2[1] - x1[1]])

    return np.vstack([x1, x1, x2, x2]), np.vstack([dx, dy, -dx, -dy])


def alphas_to_coords(S, x0, y0, alphas):
    S = np.array(S)
    alphas = np.cumsum(alphas, 1)
    x = np.zeros((alphas.shape[0], alphas.shape[1] + 1, 2))
    x[:, 0, :] = np.array([x0, y0])
    d = np.stack([S * np.cos(alphas), S * np.sin(alphas)], 2)
    x[:, 1:, :] = d
    x = np.cumsum(x, 1)
    return x, d


# TODO: optimize if needed
def rectangle_penalty(s1, x1, s2, x2, rectangle_x, rectangle_d, max_h):
    if s1 == s2:
        return np.abs(np.sum(x2 - x1)) * max_h / 2
    elif s1 % 2 == s2 % 2:
        h = np.abs(rectangle_d[(s1 + 1) % 4, :].sum())
        s = min(s1, s2)
        a = np.abs(x1[s] - rectangle_x[0, s])
        b = np.abs(x2[s] - rectangle_x[0, s])
        S = (a + b) * h / 2
        S_total = rectangle_d[0, 0] * rectangle_d[1, 1]
        return min(S, S_total - S)
    else:
        x = np.abs(x2 - x1)
        return x[0] * x[1] / 2


def rectangle_total_penalty(population_x, population_d, rectangle_x, rectangle_d):
    n, d = population_d.shape[:2]
    intersections = intersect(population_x[:, :-1, :], population_d, rectangle_x, rectangle_d)
    does_intersect = np.any(~np.isnan(intersections), 3)
    penalty = np.zeros(n)
    for t in range(n):
        intersection_side = None
        intersection_point = None
        max_h = 0.
        for i in range(d):
            if intersection_side is not None:
                s = (intersection_side + 1) % 2
                max_h = max(max_h, np.abs(population_x[t, i, s] - intersection_point[s]))
            for j in range(4):
                if does_intersect[t, i, j]:
                    j_point = population_x[t, i] + population_d[t, i] * intersections[t, i, j, 0]
                    if intersection_side is not None:
                        penalty[t] += rectangle_penalty(intersection_side, intersection_point, j, j_point, rectangle_x,
                                                        rectangle_d, max_h)
                        intersection_side = None
                    else:
                        intersection_side = j
                        intersection_point = j_point
                        max_h = 0.
        if intersection_side is not None:
            penalty[t] += rectangle_d[0, 0] * rectangle_d[1, 1]
    return penalty


def inverse_kinematics_objective(S, target, rectangles):
    rectangle_segments = [make_rectangle(*rect) for rect in rectangles]
    target = np.array(target)

    def objective(population):
        population_x, population_d = alphas_to_coords(S, 0, 0, population)
        score = np.sum((population_x[:, -1, :] - target) ** 2, 1)
        results = [score]

        for rectangle_x, rectangle_d in rectangle_segments:
            rectangle_penalties = rectangle_total_penalty(population_x, population_d, rectangle_x, rectangle_d)
            results.append(rectangle_penalties)

        results = np.stack(results, 1)
        results[:, 1:] = np.maximum(results[:, 1:], 0.)
        return results

    return objective


def dynamic_inverse_kinematics_objectives(S, target, target_v, rectangles, rectangle_vs):
    rectangle_segments = [make_rectangle(*rect) for rect in rectangles]
    rectangle_vs = [np.array(v) for v in rectangle_vs]
    target = np.array(target)
    target_v = np.array(target_v)

    def objective(t, population):
        population_x, population_d = alphas_to_coords(S, 0, 0, population)
        score = np.sum((population_x[:, -1, :] - (target + t * target_v)) ** 2, 1)
        results = [score]

        for (rectangle_x, rectangle_d), rectangle_v in zip(rectangle_segments, rectangle_vs):
            rectangle_x = rectangle_x + t * rectangle_v
            rectangle_penalties = rectangle_total_penalty(population_x, population_d, rectangle_x, rectangle_d)
            results.append(rectangle_penalties)

        results = np.stack(results, 1)
        results[:, 1:] = np.maximum(results[:, 1:], 0.)
        return results

    return objective


def draw_solutions(alphas, S, target_x, target_y, rectangles, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    x, d = alphas_to_coords(S, 0, 0, alphas)
    for rect in rectangles:
        rect_x, rect_d = make_rectangle(*rect)
        rect_patch = patches.Rectangle(rect_x[0, :], rect_d[0, 0], rect_d[1, 1], edgecolor="k", facecolor=(0, 0, 0, 0),
                                       linewidth=3)
        ax.add_patch(rect_patch)

    plt.scatter(target_x, target_y, c="red", s=400)
    ax.set_aspect("equal", "box")
    for specimen_x in x:
        ax.plot(specimen_x[:, 0], specimen_x[:, 1], marker='D')
