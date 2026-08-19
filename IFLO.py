import math
import time
import numpy as np


# Improved Frilled Lizard Optimization (IFLO)
# Update starts from LineNo : 44
def IFLO(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape[0], X.shape[1]

    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)
    ct = time.time()
    for t in range(Max_iterations):  # algorithm iteration
        # update: BEST proposed solution
        Fbest = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            xbest = X[blocation, :]  # Optimal location
            fbest = Fbest  # The optimization objective function
        elif Fbest < fbest:
            fbest = Fbest
            xbest = X[blocation, :]

        for i in range(SearchAgents):
            # Phase 1: Hunting strategy (exploration) candidate_preys
            prey_position = np.where(fit < fit[i])[0]  # Eq(4)
            if prey_position.size == 0:
                selected_prey = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_prey = xbest
                else:
                    k = np.random.choice(prey_position)
                    selected_prey = X[k, :]

            # Objective Function Formula
            r = round(1 + np.random.rand())
            rand_value = fitness[i] / (np.max(fitness[i]) ** 2 + np.min(fitness[i]) + math.sqrt(fitness[i]))
            X_new_P1 = X[i, :] + rand_value * (selected_prey - r * X[i, :])  # Eq(5)
            X_new_P1 = np.maximum(X_new_P1, lowerbound)
            X_new_P1 = np.minimum(X_new_P1, upperbound)

            # update position based on Eq (6)
            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1[i] < fit[i]:
                X[i, :] = X_new_P1[i, :]
                fit[i] = fit_new_P1[i]

            # Phase 2: Moving up the tree (exploitation)
            X_new_P2 = X[i, :] + (1 - 2 * np.random.rand()) * ((upperbound - lowerbound) / (t + 1))  # Eq(7)
            X_new_P2 = np.maximum(X_new_P2, lowerbound / (t + 1))
            X_new_P2 = np.minimum(X_new_P2, upperbound / (t + 1))

            # update position based on Eq (8)
            L = X_new_P2
            fit_new_P2 = fitness(L)
            if fit_new_P2[i] < fit[i]:
                X[i, :] = X_new_P2[i, :]
                fit[i] = fit_new_P2[i]

        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = xbest
    FLO_curve = best_so_far
    ct = time.time() - ct
    return Best_score, FLO_curve, Best_pos, ct
