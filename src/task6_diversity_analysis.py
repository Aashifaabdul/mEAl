import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from itertools import product
from scipy import stats
import sys
import os


results_dir="output"
# Import functions from other task files
from src.task1_data_setup import *
from src.task2_algorithm_setup import *
def jaccard_distance(meal1, meal2):
    """Calculates Jaccard Distance (1 - Jaccard Index) between two meals based on food items."""
    foods1 = set(f for f, q in meal1)
    foods2 = set(f for f, q in meal2)
    
    # Handle empty sets to prevent ZeroDivisionError
    if not foods1 and not foods2:
        return 0.0 # Same and empty
    if not foods1 or not foods2:
        return 1.0 # Completely different if one is empty

    intersection = len(foods1.intersection(foods2))
    union = len(foods1.union(foods2))
    
    return 1.0 - (intersection / union)

def run_ga_baseline(df, thresholds, p):
    """Baseline GA using Truncation Selection."""
    pop_size = p['pop_size']
    generations = p['generations']
    
    population = create_random_population(df, size=pop_size, max_foods=p['max_foods'])
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]

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
            if len(parents) < 2: parents.append(random.choice(population)[0])
            child = set_based_crossover(random.choice(parents), random.choice(parents), max_foods=p['max_foods'])
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
        
        # 1. Select the entire current population as parents
        parents = [meal for meal, _ in evaluated]
        best_fitness_this_run = max(best_fitness_this_run, max(fit for _, fit in evaluated))
        
        # 2. Generate an equal number of children (one-to-one mapping)
        children = []
        for i in range(pop_size // 2):
            # Select two distinct parents randomly
            p1_idx, p2_idx = random.sample(range(pop_size), 2)
            p1, p2 = parents[p1_idx], parents[p2_idx]
            
            # Crossover (produces one child for simplicity/one-to-one crowding)
            c1 = set_based_crossover(p1, p2, max_foods=p['max_foods'])
            c1 = mutate(c1, mutation_rate=p['mutation_rate'])
            
            # We will use c1 only for the one-to-one replacement
            children.append(c1)
        
        # Evaluate children
        evaluated_children = [(meal, fitness_method(meal, df, thresholds)) for meal in children if meal]
        
        new_evaluated = list(evaluated) # Copy of the current generation
        
        # 3. Crowding Replacement: Child competes with *most similar* parent
        # Note: We must pair children with parents that are *not* used by other children.
        # This implementation uses the first pop_size//2 parents/children generated above for simplicity.
        
        for i in range(len(children)):
            child_meal, child_fit = evaluated_children[i]
            
            # Find the most similar parent (among p1_idx and p2_idx used to create it)
            # For simplicity, we compare the child against ALL existing individuals in the current population (new_evaluated)
            
            # Find the closest matching meal in the current population
            closest_idx = -1
            min_distance = float('inf')
            
            for idx, (parent_meal, parent_fit) in enumerate(new_evaluated):
                dist = jaccard_distance(child_meal, parent_meal)
                if dist < min_distance:
                    min_distance = dist
                    closest_idx = idx
            
            if closest_idx != -1:
                # Deterministic Crowding: If child is better than closest parent, replace parent.
                parent_meal, parent_fit = new_evaluated[closest_idx]
                if child_fit > parent_fit:
                    new_evaluated[closest_idx] = (child_meal, child_fit)
                    
        evaluated = new_evaluated

    evaluated.sort(key=lambda x: x[1], reverse=True)
    if evaluated:
        return max(best_fitness_this_run, evaluated[0][1])
    return best_fitness_this_run
def run_statistical_comparison_diversity(n_runs=30):
    """
    Simple statistical comparison with Welch's T-Test
    """
    
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("Baseline GA (mealPlanner) vs Deterministic Crowding")
    print("="*70)
    
    # Load data
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    params = {
        'pop_size': 100,
        'generations': 10,
        'max_foods': 4,
        'mutation_rate': 0.2,
        'truncation_pct': 0.4,
        'niche_radius': 0.5
    }
    
    # Hypotheses
    print("\nHYPOTHESES:")
    print("  H₀ (Null):        μ_crowding ≤ μ_baseline")
    print("  H₁ (Alternative): μ_crowding > μ_baseline")
    print("  α (Alpha):        0.05")
    
    # Run experiments
    print(f"\nRunning {n_runs} experiments for each algorithm...")
    
    baseline_results = []
    crowding_results = []
    
    for run in range(1, n_runs + 1):
        baseline_fitness = run_ga_baseline(df, thresholds, params)
        baseline_results.append(baseline_fitness)
        
        crowding_fitness = run_ga_crowding(df, thresholds, params)
        crowding_results.append(crowding_fitness)
        
        if run % 10 == 0:
            print(f"  Completed {run}/{n_runs} runs")
    
    # Statistics
    baseline_mean = np.mean(baseline_results)
    crowding_mean = np.mean(crowding_results)
    
    print("\nRESULTS:")
    print(f"  Baseline GA Mean:          {baseline_mean:.2f}")
    print(f"  Deterministic Crowding Mean: {crowding_mean:.2f}")
    print(f"  Difference:                  {crowding_mean - baseline_mean:.2f}")
    
    # Welch's T-Test
    t_statistic, p_value_two_tailed = stats.ttest_ind(
        crowding_results, 
        baseline_results, 
        equal_var=False
    )
    
    p_value = p_value_two_tailed / 2 if t_statistic > 0 else 1 - (p_value_two_tailed / 2)
    
    print("\nWELCH'S T-TEST:")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value:     {p_value:.6f}")
    
    # Conclusion
    alpha = 0.05
    print("\nCONCLUSION:")
    if p_value < alpha:
        print(f"   REJECT H₀ (p = {p_value:.6f} < {alpha})")
        print(f"  Deterministic Crowding significantly outperforms Baseline GA")
    else:
        print(f"   FAIL TO REJECT H₀ (p = {p_value:.6f} ≥ {alpha})")
        print(f"  No significant difference detected")
    
    print("="*70 + "\n")
    
    # Save results
    results_df = pd.DataFrame({
        'Run': range(1, n_runs + 1),
        'Baseline_GA': baseline_results,
        'Deterministic_Crowding': crowding_results
    })
    output_path = os.path.join(results_dir,'task6_comparison_results.csv')
    results_df.to_csv(output_path, index=False) 
    
    return {
        'baseline_mean': baseline_mean,
        'crowding_mean': crowding_mean,
        't_statistic': t_statistic,
        'p_value': p_value,
        'reject_null': p_value < alpha
    }



