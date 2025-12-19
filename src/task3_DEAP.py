"""
DEAP implementation with fitness tracking for comparison
"""

from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
from src.task1_data_setup import *
from src.task2_algorithm_setup import *

# 1. Define Fitness Type
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 2. Define Individual Type
creator.create("Individual", list, fitness=creator.FitnessMax)

# SETUP THE TOOLBOX 

def init_individual(df, max_foods):
    """Initializes a DEAP Individual using your custom meal creation logic."""
    meal= create_random_population(df, size=1, max_foods=max_foods)[0]
    return creator.Individual(meal)

df, train_data, test_data = load_and_prepare_data()
df = add_nutrient_metrics(df)
thresholds = define_thresholds(df)

# Initialize the Toolbox
toolbox = base.Toolbox()

# Register the factory for individuals (meals)
toolbox.register("individual", init_individual, df=df, max_foods=4) 

# Register the factory for the entire population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function (Fitness)
def eval_meal(individual, df, thresholds):
    return fitness_method(individual, df, thresholds),

toolbox.register("evaluate", eval_meal, df=df, thresholds=thresholds)

# Register Selection
toolbox.register("select", tools.selTournament, tournsize=3)

# Register Crossover
toolbox.register("mate", set_based_crossover, max_foods=4) 

# Register Mutation
def mut_meal(individual):
    return mutate(individual),

toolbox.register("mutate", mut_meal)

def deap_evolve(df, thresholds, NGEN=10, CXPB=0.7, MUTPB=0.3):
    """
    Run DEAP evolution and track fitness history.
    Returns:
        best_individual: Best meal found
        best_fitness_history: List of best fitness per generation
        avg_fitness_history: List of average fitness per generation
    """
    # Create the initial population
    population = toolbox.population(n=100)
    
    # Track fitness history
    best_fitness_history = []
    avg_fitness_history = []
    
    # Hall of fame to track best individuals
    hof = tools.HallOfFame(1)
    
    # Evaluate the entire population (first generation)
    fitnesses = list(map(toolbox.evaluate, population))
    #print(fitnesses)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    for gen in range(NGEN):
        # Track statistics for this generation
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness_history.append(max(fits))
        avg_fitness_history.append(np.mean(fits))
        
        print(f"  Gen {gen+1:2d}: Best={max(fits):5.1f}, Avg={np.mean(fits):5.1f}")
        
        # 1. Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # 2. Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # 3. Apply Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        # 4. Apply Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 5. Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 6. Replace the current population
        population[:] = offspring
        
        # Update hall of fame
        hof.update(population)
        
    # Return best individual and fitness histories
    best_ind = hof[0]
    return best_ind, best_fitness_history, avg_fitness_history