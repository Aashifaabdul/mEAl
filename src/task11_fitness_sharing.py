import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from itertools import product
from scipy import stats
import sys
import os
from src.task1_data_setup import *
from src.task2_algorithm_setup import *

def jaccard_distance(meal1, meal2):
    """Calculates Jaccard Distance (1 - Jaccard Index) between two meals based on food items."""
    foods1 = set(f for f, q in meal1)
    foods2 = set(f for f, q in meal2)

    # Handle empty sets to prevent ZeroDivisionError
    if not foods1 and not foods2:
        return 0.0  # Same and empty
    if not foods1 or not foods2:
        return 1.0  # Completely different if one is empty

    intersection = len(foods1.intersection(foods2))
    union = len(foods1.union(foods2))

    return 1.0 - (intersection / union)


def run_ga_baseline(df, thresholds, p):
    """Baseline GA using Truncation Selection."""
    pop_size = p['pop_size']
    generations = p['generations']

    population = create_random_population(df, size=pop_size, max_foods=p['max_foods'])
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population if meal]

    best_fitness_this_run = -float('inf')

    for _ in range(generations):
        # 1. Selection
        evaluated.sort(key=lambda x: x[1], reverse=True)
        best_fitness_this_run = max(best_fitness_this_run, evaluated[0][1])
        selected = truncation_selection(evaluated, p['truncation_pct'])

        if not selected:
            population = create_random_population(df, size=pop_size, max_foods=p['max_foods'])
            evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population if meal]
            continue

        parents = [m for m, _ in selected]
        new_gen = []

        # 2. Crossover and Mutation
        num_children = pop_size - len(parents)
        for _ in range(num_children):
            if len(parents) < 2:
                # If not enough parents, pick a random meal from the current population to ensure 2 parents
                # Note: This pick an already selected parent, but ensures the loop continues
                parents.append(random.choice(population)[0])
            
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            child = set_based_crossover(p1, p2, max_foods=p['max_foods'])
            child = mutate(child, mutation_rate=p['mutation_rate'])
            new_gen.append(child)

        # 3. Evaluate and Replace
        next_population = parents + new_gen
        evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in next_population if meal]

    evaluated.sort(key=lambda x: x[1], reverse=True)
    if evaluated:
        return max(best_fitness_this_run, evaluated[0][1])
    return best_fitness_this_run


def run_ga_crowding(df, thresholds, p):
    """Diversity GA using Deterministic Crowding (Genotypic Diversity)."""
    pop_size = p['pop_size']
    generations = p['generations']

    population = create_random_population(df, size=pop_size, max_foods=p['max_foods'])
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]

    best_fitness_this_run = -float('inf')

    for _ in range(generations):
        # Update best fitness from current generation
        current_best_fit = max(fit for _, fit in evaluated)
        best_fitness_this_run = max(best_fitness_this_run, current_best_fit)

        # 1. Parents for children generation (entire population)
        parents = [meal for meal, _ in evaluated]

        # 2. Generate an equal number of children (pop_size) using random pairings
        children = []
        for _ in range(pop_size // 2): # Generate pop_size/2 pairs, which will result in pop_size children (if set_based_crossover produced 2)
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            # Crossover always produces 2 children
            c1= set_based_crossover(p1, p2, max_foods=p['max_foods']) 
            #print (c1)
            
            c1 = mutate(c1, mutation_rate=p['mutation_rate'])
            #c2 = mutate(c2, mutation_rate=p['mutation_rate'])
            
            children.extend([c1])
        
        # Ensure children count matches pop_size
        children = children[:pop_size]

        # Evaluate children
        evaluated_children = [(meal, fitness_method(meal, df, thresholds)) for meal in children if meal]
        # Ensure that evaluated_children has the same length as the original population to allow 1:1 replacement
        if len(evaluated_children) < pop_size:
            # For this example, we'll assume valid children are always generated
            pass 

        new_evaluated_population = list(evaluated) # Start with a copy of the current population

        # 3. Crowding Replacement: Each child competes with its most similar individual in the *current* population
        # To simplify, we iterate through children and find their closest match in the *previous* generation.
        # This implementation ensures that each child has a chance to replace one individual.
        
        # Keep track of which individuals in the current population have already been "challenged" and potentially replaced
        replaced_indices = set() 
        
        for child_meal, child_fit in evaluated_children:
            closest_idx = -1
            min_distance = float('inf')

            # Find the closest parent in the original population that hasn't been replaced yet
            for idx, (parent_meal, parent_fit) in enumerate(evaluated): # Compare against original parents
                # Only consider parents that haven't been successfully replaced by another child in this generation's crowding
                if idx in replaced_indices:
                    continue
                
                dist = jaccard_distance(child_meal, parent_meal)
                if dist < min_distance:
                    min_distance = dist
                    closest_idx = idx
            
            # If a closest parent is found and the child is better, replace the parent
            if closest_idx != -1:
                parent_meal, parent_fit = evaluated[closest_idx] # Get details from original evaluated
                if child_fit > parent_fit:
                    new_evaluated_population[closest_idx] = (child_meal, child_fit)
                    replaced_indices.add(closest_idx) # Mark this index as replaced

        evaluated = new_evaluated_population # Update the population for the next generation

    # Final sort to get the best fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)
    if evaluated:
        return max(best_fitness_this_run, evaluated[0][1])
    return best_fitness_this_run


def run_ga_fitness_sharing(df, thresholds, p):
    """
    Diversity GA using Fitness Sharing.
    
    Fitness Sharing reduces the fitness of individuals in crowded regions
    by sharing it among similar individuals, encouraging population spread.
    """
    pop_size = p['pop_size']
    generations = p['generations']
    niche_radius = p.get('niche_radius', 0.5)  # Similarity threshold
    alpha = p.get('alpha', 1.0)  # Sharing function parameter

    population = create_random_population(df, size=pop_size, max_foods=p['max_foods'])
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]
    
    best_fitness_this_run = -float('inf')

    for _ in range(generations):
        # Update best fitness from current raw fitness
        current_best_raw = max(fit for _, fit in evaluated)
        best_fitness_this_run = max(best_fitness_this_run, current_best_raw)

        # Calculate shared fitness for each individual
        shared_fitness_evaluated = [] # Stores (meal, raw_fitness, shared_fitness)
        
        for i, (meal_i, raw_fitness_i) in enumerate(evaluated):
            # Calculate niche count (sum of sharing function values)
            niche_count = 0.0
            
            for j, (meal_j, raw_fitness_j) in enumerate(evaluated):
                # Calculate distance between individuals
                distance = jaccard_distance(meal_i, meal_j)
                
                # Sharing function
                if distance < niche_radius:
                    sh = 1.0 - (distance / niche_radius) ** alpha
                else:
                    sh = 0.0
                
                niche_count += sh
            
            # Shared fitness = raw fitness / niche count
            # Ensure niche_count is never zero to prevent division by zero,
            # though it should be at least 1 (for the individual itself if dist=0)
            if niche_count > 0:
                shared_fit = raw_fitness_i / niche_count
            else:
                shared_fit = raw_fitness_i # Fallback, should not happen if individual is in its own niche
            
            shared_fitness_evaluated.append((meal_i, raw_fitness_i, shared_fit))
        
        # Selection based on SHARED fitness
        shared_fitness_evaluated.sort(key=lambda x: x[2], reverse=True) # Sort by shared fitness
        
        num_selected = int(pop_size * p['truncation_pct'])
        selected = shared_fitness_evaluated[:num_selected] # Truncation selection on shared fitness
        
        if not selected: # Handle rare case of empty selection
            population = create_random_population(df, size=pop_size, max_foods=p['max_foods'])
            evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population if meal]
            continue

        parents = [m for m, _, _ in selected]
        new_gen = []

        # Crossover and Mutation
        num_children = pop_size - len(parents)
        for _ in range(num_children):
            if len(parents) < 2: 
                parents.append(random.choice(shared_fitness_evaluated)[0]) # Add a random meal from shared_fitness_evaluated
            
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            child = set_based_crossover(p1, p2, max_foods=p['max_foods'])
            child = mutate(child, mutation_rate=p['mutation_rate'])
            new_gen.append(child)

        # Evaluate next generation (using raw fitness for the next iteration's shared fitness calculation)
        next_population = parents + new_gen
        evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in next_population if meal]

    # After all generations, find the final best raw fitness
    final_best_raw_fitness = -float('inf')
    if evaluated:
        final_best_raw_fitness = max(fit for _, fit in evaluated)
    
    return max(best_fitness_this_run, final_best_raw_fitness)


def run_statistical_comparison_diversity_fitshar(n_runs=30):
    """
    Simple statistical comparison with Welch's T-Test
    comparing Baseline GA, Deterministic Crowding, and Fitness Sharing.
    """
    
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON OF GA TECHNIQUES")
    print("Baseline GA vs Deterministic Crowding vs Fitness Sharing")
    print("="*70)
    
    # Load data
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    # Parameters for all algorithms
    # Niche radius is now part of the default parameters for Fitness Sharing
    # For Crowding, it is used for similarity comparison for replacement.
    params = {
        'pop_size': 100,
        'generations': 10,
        'max_foods': 4,
        'mutation_rate': 0.2,
        'truncation_pct': 0.4, # Used by Baseline and Fitness Sharing
        'niche_radius': 0.5, # Used by Crowding and Fitness Sharing
    }
    
    print("\nCommon Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Hypotheses (for one-tailed tests of superiority)
    print("\nHYPOTHESES (One-tailed Welch's T-Test for superiority, α=0.05):")
    print("  H₀: μ_A ≤ μ_B (A is not significantly better than B)")
    print("  H₁: μ_A > μ_B (A is significantly better than B)")
    
    # Run experiments
    print(f"\nRunning {n_runs} experiments for each algorithm...")
    
    baseline_results = []
    crowding_results = []
    sharing_results = []
    
    for run in range(1, n_runs + 1):
        baseline_fitness = run_ga_baseline(df, thresholds, params)
        baseline_results.append(baseline_fitness)
        
        crowding_fitness = run_ga_crowding(df, thresholds, params)
        crowding_results.append(crowding_fitness)

        sharing_fitness = run_ga_fitness_sharing(df, thresholds, params)
        sharing_results.append(sharing_fitness)
        
        if run % 5 == 0: # Print progress every 5 runs
            print(f"  Completed {run}/{n_runs} runs")
    
    # Store all raw results in a DataFrame
    raw_results_df = pd.DataFrame({
        'Run': range(1, n_runs + 1),
        'Baseline_GA': baseline_results,
        'Deterministic_Crowding': crowding_results,
        'Fitness_Sharing': sharing_results
    })
    results_dir="output"
    output_path = os.path.join(results_dir, 'task11_raw_comparison_results.csv')

    raw_results_df.to_csv(output_path, index=False)

    # Statistics
    baseline_mean = np.mean(baseline_results)
    crowding_mean = np.mean(crowding_results)
    sharing_mean = np.mean(sharing_results)
    
    print("\nMEAN FITNESS RESULTS (Average of Best Fitness per Run):")
    print(f"  Baseline GA:             {baseline_mean:.2f}")
    print(f"  Deterministic Crowding:  {crowding_mean:.2f}")
    print(f"  Fitness Sharing:         {sharing_mean:.2f}")
    
    # Perform Welch's T-Tests for pairwise comparison
    alpha = 0.05
    print("\n====================================")
    print("STATISTICAL TESTS (Welch's T-Test)")
    print("======================================")

    test_results = []

    # Helper function for one-tailed test
    def perform_one_tailed_welch_t_test(data_a, data_b, name_a, name_b):
        # H0: mu_a <= mu_b, H1: mu_a > mu_b
        t_statistic, p_value_two_tailed = stats.ttest_ind(data_a, data_b, equal_var=False)
        
        if t_statistic > 0:
            p_value = p_value_two_tailed / 2
        else:
            p_value = 1 - (p_value_two_tailed / 2) # If t is negative, p for H1 is high

        conclusion = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"
        significant = " SIGNIFICANT DIFFERENCE" if p_value < alpha else " NO SIGNIFICANT DIFFERENCE"
    
        print(f"\n[Test] {name_a} vs {name_b}")
        print(f"  H₀: μ_{name_a} ≤ μ_{name_b}")
        print(f"  H₁: μ_{name_a} > μ_{name_b}")
        print(f"  t-statistic: {t_statistic:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        print(f"  {conclusion} (p {'<' if p_value < alpha else '≥'} {alpha}) - {significant}")
        return {'alg1': name_a, 'alg2': name_b, 't_stat': t_statistic, 'p_val': p_value, 'conclusion': conclusion, 'significant': p_value < alpha}


    # Test 1: Crowding vs Baseline
    test_results.append(perform_one_tailed_welch_t_test(crowding_results, baseline_results, "Crowding", "Baseline GA"))

    # Test 2: Sharing vs Baseline
    test_results.append(perform_one_tailed_welch_t_test(sharing_results, baseline_results, "Sharing", "Baseline GA"))
    
    # Test 3: Sharing vs Crowding
    test_results.append(perform_one_tailed_welch_t_test(sharing_results, crowding_results, "Sharing", "Crowding"))


    print("\n===========================================")
    print("OVERALL RANKING (Based on Mean Fitness)")
    print("=============================================")
    
    ranking_data = [
        ("Baseline GA", baseline_mean),
        ("Deterministic Crowding", crowding_mean),
        ("Fitness Sharing", sharing_mean)
    ]
    
    ranking_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, mean_fit) in enumerate(ranking_data):
        print(f"  {i+1}. {name:25s} - {mean_fit:.2f}")
    
    best_technique_name = ranking_data[0][0]
    print(f"\n BEST TECHNIQUE: {best_technique_name}")
    print("="*70 + "\n")
    
    return {
        'baseline_mean': baseline_mean,
        'crowding_mean': crowding_mean,
        'sharing_mean': sharing_mean,
        'tests': test_results,
        'overall_best': best_technique_name
    }

