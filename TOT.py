import time
import numpy as np


def TOT(X, fitness, lowerbound, upperbound, Max_iterations, minimize=True, seed=None):
    rng = np.random.default_rng(seed)
    n_agents, dim = X.shape
    lowerbound = np.asarray(lowerbound)
    upperbound = np.asarray(upperbound)
    span = upperbound - lowerbound

    # Evaluate initial population
    fit = np.array([fitness(x) for x in X])
    if not minimize:
        fit = -fit

    Best_score = np.min(fit)
    Best_pos = X[np.argmin(fit)].copy()
    TOTO_curve = np.zeros(Max_iterations)
    start_time = time.time()

    for t in range(Max_iterations):
        shrink_factor = 1.0 - t / Max_iterations  # local shrink

        for i in range(n_agents):
            x = X[i].copy()
            x_val = fit[i]

            # pick a random agent j != i
            j = i
            while j == i and n_agents > 1:
                j = rng.integers(0, n_agents)
            s_agent = X[j].copy()
            s_val = fit[j]

            r1 = rng.random()
            r3 = rng.random() * 2 - 1.0

            # ---- Three guided searches ----
            c1 = x + r1 * (Best_pos - x) * (1.0 + 0.5 * r3)
            c2 = Best_pos + r1 * (Best_pos - x) * (0.5 + r3)
            if s_val < x_val:
                c3 = x + r1 * (s_agent - x) * (1.0 + 0.3 * r3)
            else:
                c3 = x + r1 * (x - s_agent) * (1.0 + 0.3 * r3)

            # ---- Three random/local searches ----
            local_radius = 0.5 * span * shrink_factor
            c4 = x + (rng.random(dim) * 2 - 1.0) * local_radius[i]
            c5 = Best_pos + (rng.random(dim) * 2 - 1.0) * local_radius[i]
            c6 = rng.random(dim) * span + lowerbound[i]

            # Clip candidates
            candidates = np.clip(np.vstack([c1, c2, c3, c4, c5, c6]), lowerbound[i], upperbound[i])

            # Evaluate candidates
            cand_vals = np.array([fitness(c) for c in candidates])
            if not minimize:
                cand_vals = -cand_vals

            best_c_idx = np.argmin(cand_vals)
            best_c = candidates[best_c_idx]
            best_c_val = cand_vals[best_c_idx]

            # Accept if better
            if best_c_val < x_val:
                X[i] = best_c
                fit[i] = best_c_val
                x_val = best_c_val

            # Update global best
            if x_val < Best_score:
                Best_score = x_val
                Best_pos = X[i].copy()

        TOTO_curve[t] = Best_score

    elapsed_time = time.time() - start_time
    if not minimize:
        Best_score = -Best_score
        TOTO_curve = -TOTO_curve

    return Best_score, TOTO_curve, Best_pos, elapsed_time
