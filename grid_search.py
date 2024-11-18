from ant_colony_opt import *

def grid_search_hyperparams():
    """tuning 3 hyperparameters, definition and further investigation in report.pdf"""
    alpha_values = [1.3]  # Example values for alpha
    beta_values = [5.0]   # Example values for beta
    decay_values = [0.89, 0.9, 0.91, 0.95]

    best_fitness = 0
    best_params = None

    # perform grid search over all hyperparam combinations
    for alpha in alpha_values:
        for beta in beta_values:
            for decay in decay_values:
                solution = AnyColonySolution(n=100, alpha=alpha, beta=beta, decay=decay)
                fitness = solution.run_algo()
            
                print(f"Alpha: {alpha}, Beta: {beta}, Decay: {decay} -> Fitness: {fitness}")
            
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = {'alpha': alpha, 'beta': beta, 'decay': decay}

    print("\nBest fitness found:", best_fitness)
    print("Best parameters:", best_params)


def grid_search_population_size():
    """
    search for optimal population size (number of ants running in parallel)
    in each iteration, balance run time and performance
    """
    n_ants_val = [1,10,50,100,200]
    for n_ant in n_ants_val:
        solution = AnyColonySolution(n=50, n_ants=n_ant, alpha=1.3, beta=5.0, decay=0.9)
        fitness = solution.run_algo()
            
        print(f"Number of ants: {n_ant} -> Fitness: {fitness}")
            
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = {'n_ants': n_ant}

    print("\nBest fitness found:", best_fitness)
    print("Best parameters:", best_params)