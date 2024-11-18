from ant_colony_opt import *

################################################################
# To Run The File: In terminal, go to current file directory (./NIC_code)
# Paste "python ant_colony_opt.py"
# Return example: Best fitness found: 4520
################################################################
def main():
    #solution = AnyColonySolution(n=10000, n_ants=10, alpha=1.3, beta=5.0, decay=0.9)
    #although coursework suggests 10,000 iterations, run time too long 
    #best fitness after 10,000 is 4528, parameters are optimised using grid search
    solution = AnyColonySolution(n=10, n_ants=10, alpha=1.3, beta=5.0, decay=0.9)
    best_fitness = solution.run_algo()
    print("Best fitness found:", best_fitness)

if __name__ == "__main__":
    main()