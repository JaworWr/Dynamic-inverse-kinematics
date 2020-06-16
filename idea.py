import numpy as np


def FNS(scores):
    domination = np.all(scores[:, None, :] <= scores[None, :, :], axis=2)  # domination[i, j] = "i dominuje j"
    domination &= np.any(scores[:, None, :] < scores[None, :, :], axis=2)
    Nx = domination.sum(0)

    Pf = []
    ranks = np.zeros(scores.shape[0])
    r = 0
    Q = np.nonzero(Nx == 0)[0]
    while Q.size > 0:
        Nx[Q] = -1
        Pf.append(Q)
        ranks[Q] = r
        r += 1
        for i in Q:
            Nx[domination[i, :]] -= 1
        Q = np.nonzero(Nx == 0)[0]

    return Pf, ranks


def crowding_distance(scores):
    indices = np.argsort(scores, 0)
    sorted_scores = np.take_along_axis(scores, indices, 0)
    cd = np.zeros(scores.shape[0])
    for k in range(scores.shape[1]):
        if sorted_scores[-1, k] != sorted_scores[0, k]:
            cd[indices[[0, -1], k]] = np.inf
            cd[indices[1:-1, k]] += (sorted_scores[2:, k] - sorted_scores[:-2, k]) / (
                        sorted_scores[-1, k] - sorted_scores[0, k])
    return cd


def random_population(d, n, x_min, x_max):
    return np.hstack([np.random.uniform(x_min, x_max, (n, d))])


def tournament_selection(ranks, dists, n):
    candidates = np.random.choice(n, (n, 2), replace=True)
    mask = np.where(
        ranks[candidates[:, 0]] == ranks[candidates[:, 1]],
        dists[candidates[:, 0]] > dists[candidates[:, 1]],
        ranks[candidates[:, 0]] < ranks[candidates[:, 1]]
    )
    result = candidates[:, 1]
    result[mask] = candidates[mask, 0]
    return result


def crossover(x, p, eta):  # simulated binary crossover
    n, d = x.shape
    l = n // 2
    mask = np.random.random(l) <= p
    m = np.sum(mask)
    mi = np.random.random((m, d))
    beta = np.where(
        mi < 0.5,
        np.power(2 * mi, 1. / (eta + 1.)),
        np.power(1. / (2. * (1 - mi)), 1. / (eta + 1.))
    )
    c1 = x[:l, :].copy()
    c2 = x[l:, :].copy()
    c1[mask, :] = 0.5 * (1 + beta) * x[:l, :][mask, :] + 0.5 * (1 - beta) * x[l:, :][mask, :]
    c2[mask, :] = 0.5 * (1 + beta) * x[:l, :][mask, :] + 0.5 * (1 - beta) * x[l:, :][mask, :]
    return np.vstack([c1, c2])


def mutation(x, x_min, x_max, p, eta):  # polynomial mutation
    n, d = x.shape
    mask = np.random.random(x.shape[0]) <= p
    m = np.sum(mask)
    mi = np.random.random((m, d))
    beta = np.where(
        mi < 0.5,
        np.power(2 * mi, 1. / (eta + 1.)) - 1.,
        1. - np.power(2. * (1 - mi), 1. / (eta + 1.))
    )
    y = x.copy()
    y[mask, :] = np.where(
        mi < 0.5,
        x[mask, :] + beta * (x[mask, :] - x_min),
        x[mask, :] + beta * (x_max - x[mask, :])
    )
    return y


def elitist_selection(fronts, dists, to_take):
    taken = []
    for front in fronts:
        if len(front) <= to_take:
            taken += list(front)
            if len(front) == to_take:
                break
            to_take -= len(front)
        else:
            indices = np.argsort(-dists[front])[:to_take]
            taken += list(front[indices])
            break
    return taken


def constraint_violation(constraints):
    n, d = constraints.shape
    sort_indices = np.argsort(constraints, 0)
    violations = np.zeros(n)
    for i in range(d):
        values, counts = np.unique(constraints[:, i], return_counts=True)  # unikalne wartości są zwracane posortowane
        counts = np.cumsum(counts)
        counts = list(counts)
        if values[0] != 0:
            counts = [0] + counts
        for rank, (j, k) in enumerate(zip([0] + counts, counts + [len(counts)])):
            violations[sort_indices[j:k, i]] += rank
    return violations


def evaluation(objective, n_constraints, population):
    obj_results = objective(population)
    constraint_values = obj_results[:, -n_constraints:]
    violation_measure = constraint_violation(constraint_values)
    scores = np.concatenate([obj_results[:, :-n_constraints], violation_measure[:, None]], 1)
    return scores


def IDEA(objective, n_constraints, x_min, x_max, d, n, *args, **kwargs):
    population = random_population(d, n, x_min, x_max)
    return sub_IDEA(population, objective, n_constraints, x_min, x_max, n, *args, **kwargs)


def dynamic_IDEA(objective, n_constraints, T, x_min, x_max, d, n, *args, num_iterations_init, num_iterations, n_immigrants=0, **kwargs):
    population = random_population(d, n, x_min, x_max)

    print("=" * 80)
    print("t=0")
    print("=" * 80)

    t = 0

    def round_objective(round_population):
        return objective(t, round_population)

    p, s = sub_IDEA(population, round_objective, n_constraints, x_min, x_max, n, *args,
                    num_iterations=num_iterations_init, **kwargs)
    population_history = [p]
    score_history = [s]

    for t in range(1, T):
        print("=" * 80)
        print(f"t={t}")
        print("=" * 80)

        population = p[-1, :, :]
        if n_immigrants > 0:
            immigrants = random_population(d, n_immigrants, x_min, x_max)
            population = np.vstack([population, immigrants])
        p, s = sub_IDEA(population, round_objective, n_constraints, x_min, x_max, n, *args,
                        num_iterations=num_iterations, **kwargs)
        population_history.append(p)
        score_history.append(s)

    return population_history, score_history


def sub_IDEA(population, objective, n_constraints, x_min, x_max, n, n_inf, eta_c, eta_m, p_c, p_m, num_iterations, log_interval=10):
    n_f = n - n_inf
    populations = []
    scores = evaluation(objective, n_constraints, population)
    scores_hist = []

    fronts, ranks = FNS(scores)
    dists = crowding_distance(scores)

    for iter_ in range(num_iterations):
        parent_indices = tournament_selection(ranks, dists, n)
        offspring = crossover(population[parent_indices, :], p_c, eta_c)
        offspring = np.clip(offspring, x_min, x_max)
        offspring = mutation(offspring, x_min, x_max, p_m, eta_m)
        offspring_scores = evaluation(objective, n_constraints, offspring)

        population = np.vstack([population, offspring])
        scores = np.vstack([scores, offspring_scores])

        dists = crowding_distance(scores)
        mask_f = scores[:, -1] == 0
        mask_inf = ~mask_f
        s_f = np.sum(mask_f)
        s_inf = np.sum(mask_inf)
        if s_f < n_f:
            to_take_f = s_f
            to_take_inf = n - s_f
        elif s_inf < n_inf:
            to_take_inf = s_inf
            to_take_f = n - s_inf
        else:
            to_take_f = n_f
            to_take_inf = n_inf

        population_f = population[mask_f, :]
        scores_f = scores[mask_f, :]
        dists_f = dists[mask_f]
        fronts_f, ranks_f = FNS(scores_f)
        taken_f = elitist_selection(fronts_f, dists_f, to_take_f)

        population_inf = population[mask_inf, :]
        scores_inf = scores[mask_inf, :]
        dists_inf = dists[mask_inf]
        fronts_inf, ranks_inf = FNS(scores_inf)
        taken_inf = elitist_selection(fronts_inf, dists_inf, to_take_inf)

        population = np.vstack([population_f[taken_f, :], population_inf[taken_inf, :]])
        scores = np.vstack([scores_f[taken_f, :], scores_inf[taken_inf, :]])
        dists = np.hstack([dists_f[taken_f], dists_inf[taken_inf]])
        fronts, ranks = FNS(scores)

        populations.append(population.copy())
        scores_hist.append(scores.copy())

        if iter_ % log_interval == 0:
            print(f"Iteration {iter_}, #feasible: {to_take_f}, best: {scores_f[taken_f, :-1].min(0) if to_take_f else '-'}, " +
                  f"#infeasible: {to_take_inf}, best: {scores_inf[taken_inf, :].min(0) if to_take_inf else '-'}")
    print(f"Iteration {iter_}, #feasible: {to_take_f}, best: {scores_f[taken_f, :-1].min(0) if to_take_f else '-'}, " +
          f"#infeasible: {to_take_inf}, best: {scores_inf[taken_inf, :].min(0) if to_take_inf else '-'}")
    return np.stack(populations, 0), np.stack(scores_hist, 0)
