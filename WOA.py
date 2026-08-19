import time
import numpy as np

def foraging_phase(population, fitness, fobj, N, dim, VRmin, VRmax):
    new_population = np.copy(population)
    for i in range(N):
        CFP = [population[k] for k in range(N) if fitness[k] < fitness[i] and k != i]
        if CFP:
            SFP = CFP[np.random.randint(len(CFP))]
            r = np.random.rand(dim)
            new_position = population[i] + r * (SFP - population[i])
            new_population[i] = np.clip(new_position, VRmin, VRmax)
            if fobj(new_population[i]) < fitness[i]:
                fitness[i] = fobj(new_population[i])
    return new_population, fitness


def escape_phase(population, fitness, fobj, N, dim, VRmin, VRmax, t):
    new_population = np.copy(population)
    for i in range(N):
        r = np.random.rand(dim)
        new_position = population[i] + (1 - 2 * r) * (VRmax - VRmin) / t
        new_population[i] = np.clip(new_position, VRmin, VRmax)
        if fobj(new_population[i]) < fitness[i]:
            fitness[i] = fobj(new_population[i])
    return new_population, fitness


def WOA(population, fobj, VRmin, VRmax, T):
    N, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    fitness = fobj(population)
    best_solution = population[np.argmin(fitness)]
    best_fitness = float('inf')
    Convergence_curve = np.zeros((T, 1))

    t = 0
    ct = time.time()
    for t in range(1, T + 1):
        population, fitness = foraging_phase(population, fitness, fobj, N, dim, lb, ub)
        population, fitness = escape_phase(population, fitness, fobj, N, dim, lb, ub, t)
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitness)]
        Convergence_curve[0] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[T - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct