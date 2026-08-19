import time
import numpy as np


def CWO(X, objective_function, VRmin, VRmax, max_iter):
    num_agents = X.shape[0]  # Population size
    num_dimensions = X.shape[1]  # Problem dimension
    lower_bound = VRmin[0, :]  # Variable lower limit
    upper_bound = VRmax[0, :]  # Variable upper limit

    # --------------------------------------
    # Initialization
    # --------------------------------------
    fitness = np.array([objective_function(ind) for ind in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()

    # --------------------------------------
    # Main CWO loop
    # --------------------------------------
    for t in range(max_iter):
        # Control parameters (shrink over time)
        w = 1 - (t / max_iter)  # Weaving factor
        b = 2 * np.exp(-((4 * t) / max_iter) ** 2)  # Spiral shrink

        for i in range(num_agents):
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Equation 1: Weaving search around best solution
            spiral_radius = np.abs(r1 * (X_best - X[i]))
            theta = r2 * 2 * np.pi
            X_spiral = X_best + spiral_radius * np.cos(theta) + spiral_radius * np.sin(theta)

            # Equation 2: Random weaving adjustment
            rand_idx = np.random.randint(num_agents)
            X_rand = X[rand_idx]
            X_random = X[i] + w * (X_rand - X[i])

            # Equation 3: Combine spiral and random movement
            X_new = b * X_spiral + (1 - b) * X_random

            # Equation 4: Boundary control
            X_new = np.clip(X_new, lower_bound, upper_bound)

            # Evaluate new position
            new_fitness = objective_function(X_new)

            # Greedy update
            if new_fitness < fitness[i]:
                X[i] = X_new
                fitness[i] = new_fitness
                if new_fitness < objective_function(X_best):
                    X_best = X_new.copy()

        Convergence_curve[t] = fitness[np.argmin(fitness)]

    Leader_score = Convergence_curve[-1][0]
    ct = time.time() - ct

    return Leader_score, Convergence_curve, X_best, ct
