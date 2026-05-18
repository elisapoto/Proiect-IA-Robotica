import numpy as np
import random
import math

class TSPSolver:
    def __init__(self, cities_coords):
        """
        cities_coords: un array NumPy de formă (N, 2) cu coordonatele X și Y ale orașelor.
        """
        self.cities = cities_coords
        self.num_cities = len(cities_coords)
        self.dist_matrix = self._create_dist_matrix()

    def _create_dist_matrix(self):
        # Calculează distanțele dintre oricare două orașe (Matricea de distanțe)
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                matrix[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return matrix

    def compute_path_cost(self, path):
        # Calculează lungimea totală a unui traseu (inclusiv întoarcerea acasă)
        cost = 0
        for i in range(len(path)):
            cost += self.dist_matrix[path[i]][path[(i + 1) % len(path)]]
        return cost

    # ==========================================
    # 1. BACKTRACKING (BKT) - Atenție, maxim 10 orașe!
    # ==========================================
    def solve_bkt(self):
        best_path = []
        best_cost = float('inf')
        
        def backtrack(current_path, current_cost):
            nonlocal best_cost, best_path
            if len(current_path) == self.num_cities:
                total_cost = current_cost + self.dist_matrix[current_path[-1]][current_path[0]]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = current_path.copy()
                return

            for next_city in range(self.num_cities):
                if next_city not in current_path:
                    new_cost = current_cost + (self.dist_matrix[current_path[-1]][next_city] if current_path else 0)
                    if new_cost < best_cost: # Tăiere (Pruning) pentru eficiență
                        current_path.append(next_city)
                        backtrack(current_path, new_cost)
                        current_path.pop()

        backtrack([0], 0) # Pornim mereu din primul oraș
        return best_path, best_cost

    # ==========================================
    # 2. HILL CLIMBING (HC)
    # ==========================================
    def solve_hc(self):
        current_path = list(range(self.num_cities))
        random.shuffle(current_path)
        current_cost = self.compute_path_cost(current_path)
        
        improved = True
        while improved:
            improved = False
            for i in range(self.num_cities):
                for j in range(i + 1, self.num_cities):
                    neighbor_path = current_path.copy()
                    # Inversăm ordinea a două orașe
                    neighbor_path[i], neighbor_path[j] = neighbor_path[j], neighbor_path[i]
                    neighbor_cost = self.compute_path_cost(neighbor_path)
                    
                    if neighbor_cost < current_cost:
                        current_cost = neighbor_cost
                        current_path = neighbor_path
                        improved = True
                        break
                if improved: break
        return current_path, current_cost

    # ==========================================
    # 3. SIMULATED ANNEALING (SA)
    # ==========================================
    def solve_sa(self, init_temp=1000, cooling_rate=0.99, stop_temp=0.01):
        current_path = list(range(self.num_cities))
        random.shuffle(current_path)
        current_cost = self.compute_path_cost(current_path)
        
        best_path = current_path.copy()
        best_cost = current_cost
        T = init_temp
        
        while T > stop_temp:
            i, j = random.sample(range(self.num_cities), 2)
            neighbor_path = current_path.copy()
            neighbor_path[i], neighbor_path[j] = neighbor_path[j], neighbor_path[i]
            neighbor_cost = self.compute_path_cost(neighbor_path)
            
            delta = neighbor_cost - current_cost
            # Dacă e mai bun, îl acceptăm. Dacă e mai prost, îl acceptăm doar cu o anumită probabilitate
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_path = neighbor_path
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = current_path.copy()
            T *= cooling_rate
        return best_path, best_cost

    # ==========================================
    # 4. NEAREST NEIGHBOR (NN - Abordarea Greedy)
    # ==========================================
    def solve_nn(self):
        unvisited = list(range(1, self.num_cities))
        current_city = 0
        path = [current_city]
        
        while unvisited:
            next_city = min(unvisited, key=lambda city: self.dist_matrix[current_city][city])
            unvisited.remove(next_city)
            path.append(next_city)
            current_city = next_city
            
        return path, self.compute_path_cost(path)

    # ==========================================
    # 5. ALGORITM GENETIC (GA) - Versiune Compactă
    # ==========================================
    def solve_ga(self, pop_size=50, generations=100, mutation_rate=0.1):
        # Generăm o populație inițială de drumuri aleatorii
        population = [random.sample(range(self.num_cities), self.num_cities) for _ in range(pop_size)]
        
        for _ in range(generations):
            # Sortăm populația în funcție de performanță (cost cel mai mic)
            population = sorted(population, key=lambda p: self.compute_path_cost(p))
            
            new_population = population[:10] # Păstrăm top 10 cele mai bune soluții (Elitism)
            
            while len(new_population) < pop_size:
                # Selecție simplă: alegem doi părinți la întâmplare din top 20
                p1, p2 = random.choice(population[:20]), random.choice(population[:20])
                
                # Crossover (Combinare) tip "Ordered Crossover" simplificat
                cut = self.num_cities // 2
                child = p1[:cut]
                for city in p2:
                    if city not in child:
                        child.append(city)
                        
                # Mutație: inversăm două orașe aleatoriu
                if random.random() < mutation_rate:
                    idx1, idx2 = random.sample(range(self.num_cities), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
                    
                new_population.append(child)
            population = new_population

        best_path = min(population, key=lambda p: self.compute_path_cost(p))
        return best_path, self.compute_path_cost(best_path)