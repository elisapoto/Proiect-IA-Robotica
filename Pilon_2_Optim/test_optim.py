import numpy as np
from visuals import plot_path
from tsp_solvers import TSPSolver

# Generăm 10 orașe cu coordonate aleatorii între 0 și 100
np.random.seed(42)
coordonate_orase = np.random.rand(10, 2) * 100

# Inițializăm clasa ta
solver = TSPSolver(coordonate_orase)

print("--- TESTARE ALGORITMI TSP ---")

cale, cost = solver.solve_bkt()
print(f"BKT -> Cel mai scurt drum: {cale}, Cost total: {cost:.2f}")

cale, cost = solver.solve_nn()
print(f"NN (Greedy) -> Cel mai scurt drum: {cale}, Cost total: {cost:.2f}")

cale, cost = solver.solve_hc()
print(f"Hill Climbing -> Cel mai scurt drum: {cale}, Cost total: {cost:.2f}")

cale, cost = solver.solve_sa()
print(f"Simulated Annealing -> Cel mai scurt drum: {cale}, Cost total: {cost:.2f}")

cale, cost = solver.solve_ga()
print(f"Algoritm Genetic -> Cel mai scurt drum: {cale}, Cost total: {cost:.2f}")
plot_path(coordonate_orase, cale, "GA Solution")