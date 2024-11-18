import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock, BoundedSemaphore
import os

#import data####################################################
with open('bags.txt', 'r', encoding='utf-16') as file: 
    content = file.read()

data = []

lines = content.splitlines()

for line in lines:
    line = line.strip()
    if "w" in line:
        weight = float(line.split(":")[1].strip())
        data.append({'weight': weight, 'value': None})
    elif "v" in line:
        value = int(line.split(":")[1].strip())
        if data:
            data[-1]['value'] = value

df = pd.DataFrame(data)
################################################################
# implement the methods in ACO

################################################################
class AnyColonySolution:
    _df = df  # static dataframe, contain columns 'weight' and 'value'
    _LOCAL_HEURISTICS = _df['value'] / _df['weight'] # static, desirability = value-to-weight ratio
    _NO_OF_CPUS = os.cpu_count()  # CPU count, static for each machine
    
    def __init__(self, pheromone = None, n = 10000, n_ants=1, alpha = 1.0, beta = 1.0, decay = 0.1):
        """
        args:
            pheromone (1D numpy.array): to be updated after each iteration
            n (int): number of iterations
            n_ants (int): number of ants in each iteration
            alpha (float): hyperparam, pheromone importance. 0 = pheromone not considered
            beta (float): hyperparam, heuristic importance. 0 = heuristic not considered
            decay (float): hyperparam, rate of pheromone decay, "evaporation rate". between 0-1, higher value = faster decay
        """
        self.pheromone = np.ones(len(self._df)) if pheromone is None else pheromone # initialise pheromone as 1s
        self.n = n 
        self.n_ants = n_ants
        self.alpha = alpha 
        self.beta = beta
        self.decay = decay 

        self.best_fitness = 0
        self.lock = Lock() # avoid race condition when running in parallel


    def initialise_ant(self):
        start_node = np.random.randint(0, len(self._df)) # ant start at a random node
        tabu_list = [start_node] # memorise all visited nodes
        available_nodes = set(self._df.index) - {start_node} # since fully connected graph, remaining available nodes are all unvisited nodes
        current_weight = self._df.at[start_node, 'weight'] # keep track of weight
        current_value = self._df.at[start_node, 'value'] # keep track of value
        
        return {
            'tabu_list': tabu_list,
            'available_nodes': available_nodes,
            'current_weight': current_weight,
            'current_value': current_value,
        }


    def prob_ijt(self, ant):
        """
        calculate the probability of choosing node j (in set of all available nodes H)
        prob_j(t) = (phe_j(t) ** alpha * heuristics_j **beta) / sum of all h in H(phe_h(t) ** alpha * heuristics_h **beta)
        """
        available_nodes = list(ant['available_nodes'])
        pheromone = self.pheromone[available_nodes]
        heuristics = self._LOCAL_HEURISTICS[available_nodes]
        numerator = (pheromone ** self.alpha) * (heuristics ** self.beta)
        probabilities = numerator / numerator.sum()

        return probabilities
    

    def evaporation_rule(self):
        """
        apply evaporation after completing iteration by a factor of (1 - decay)
        """
        self.pheromone *= (1 - self.decay)
    
    def update_pheromone(self,all_tabu_lists):
        """
        deposit pheromone on all edges in tabu_list(s), after all ants complete an iteration
        phe(t+1) <- evaporation(t) + deposit increments(t) (defined below)
        """
        with self.lock:
            for tabu_list in all_tabu_lists:
                for node in tabu_list:
                    #bag_weight = self._df.at[node, 'weight']
                    bag_value = self._df.at[node, 'value']
                    #self.pheromone[node] += 1 / bag_weight
                    #self.pheromone[node] += np.log(bag_value) / bag_weight
                    self.pheromone[node] += np.log(bag_value)

    
    def evaluate_fitness(self, fitness_values):
        """
        return the max fitness considering all ant fitness levels in an iteration
        """
        return max(fitness_values)
    

    def run_ant(self):
        """
        run one ant and return its accessed nodes and fitness, this is our atomic operation
        """
        ant = self.initialise_ant()
        
        while ant['available_nodes']:
            prob_dist = self.prob_ijt(ant)
            next_node = np.random.choice(list(ant['available_nodes']), p=prob_dist)
            next_weight = self._df.at[next_node, 'weight']
            if ant['current_weight'] + next_weight > 295: # check for weight constraint
                break
            
            ant['tabu_list'].append(next_node)
            ant['available_nodes'].remove(next_node)
            ant['current_weight'] += next_weight
            ant['current_value'] += self._df.at[next_node, 'value']

        fitness = ant['current_value'] if ant['current_weight'] <= 295 else 0 # update fitness after path is completed
        return ant['tabu_list'], fitness
    

    def run_iteration(self):
        """
        in each iteration we need to memorise all tabu_lists to update pheromone
        we also need to memorise all fitness levels to assess the max fitness value
        return the best (max) fitness value in an iteration
        """
        all_tabu_lists = []
        fitness_values = []
        threads = []
        semaphore = BoundedSemaphore(self._NO_OF_CPUS)  # there're at most _NO_OF_CPUS active threads (8 for me) and at least 0 active threads

        def run_ant_thread():
            with semaphore: # when a slot is available, do an ant i:
                tabu_list, fitness = self.run_ant()
                # for ant i, append final tabu_list and fitness into our array
                all_tabu_lists.append(tabu_list) 
                fitness_values.append(fitness)

        for _ in range(self.n_ants):
            thread = Thread(target=run_ant_thread) # create "n_ants" number of threads to be executed concurrently using semaphore
            threads.append(thread)
            thread.start() # passing thread to func "run_ant_thread"

        for thread in threads:
            thread.join() # wait for all threads to complete

        max_fitness = self.evaluate_fitness(fitness_values) # 
        self.update_pheromone(all_tabu_lists)
        return max_fitness
    
    def run_algo(self):
        """
        run the ACO algorithm and return the maximum fitness considering all iterations
        """
        for _ in range(self.n):
            max_fitness = self.run_iteration() 
            self.best_fitness = max(self.best_fitness, max_fitness)
            self.evaporation_rule() 
        return self.best_fitness

