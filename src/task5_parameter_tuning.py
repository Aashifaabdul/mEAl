import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os 
from itertools import product
# Import functions from your other task files
from src.task1_data_setup import *
from src.task2_algorithm_setup import *

def run_ga_experiment(df, thresholds, pop_size, generations, selection_pct, mutation_rate, max_foods=4):
    """
    Runs a SINGLE instance of the GA with the given parameters.    
    Returns:
        best_fitness (float): The best fitness score achieved in this run.
    """
    # 1. Initialization
    population = create_random_population(df, size=pop_size, max_foods=max_foods)
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population if meal]
    if not evaluated:
        return -float('inf') # Handle empty initial population

    # Find the best fitness from the initial population
    best_fitness_this_run = max(fit for _, fit in evaluated)

    for generation in range(generations):
        # Sort and find best for this generation
        evaluated.sort(key=lambda x: x[1], reverse=True)
        
        current_best_fitness = evaluated[0][1]
        if current_best_fitness > best_fitness_this_run:
            best_fitness_this_run = current_best_fitness
        
        # 2. Selection
        selected = truncation_selection(evaluated, selection_pct)
        if not selected: # Handle rare case of empty selection
            population = create_random_population(df, size=pop_size, max_foods=max_foods)
            evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population if meal]
            if not evaluated: continue 
            continue
            
        parents = [m for m, _ in selected]
        new_gen = []

        # 3. Crossover and 4. Mutation
        num_children = len(population) - len(parents)
        for _ in range(num_children):
            # Ensure we have at least two parents to choose from
            if len(parents) < 2:
                # Add a random individual from the evaluated pool if parents are too few
                parents.append(random.choice(evaluated)[0]) 
                
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            child = set_based_crossover(p1, p2, max_foods=max_foods)
            child = mutate(child, mutation_rate=mutation_rate) # Pass parameter
            new_gen.append(child)

        # Combine parents and offspring and evaluate
        next_population = parents + new_gen
        evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in next_population if meal]
        if not evaluated:
            continue # Skip if evaluation fails
    
    # After all generations, find the final best fitness
    if evaluated:
        evaluated.sort(key=lambda x: x[1], reverse=True)
        # Final check for best fitness
        if evaluated[0][1] > best_fitness_this_run:
            best_fitness_this_run = evaluated[0][1]
        
    return best_fitness_this_run # Return best found in the run

def plot_results(results_df, param_keys, n_runs):
    """
    Visualizes the tuning results as a bar chart.
    Plots Effectiveness (Solution Quality).
    """
    # Sort by effectiveness for a cleaner plot
    results_df = results_df.sort_values(by='effectiveness_avg_fitness', ascending=False)
    
    # Create string labels for each combination
    def dict_to_str(d):
        # Shorten keys for readability
        return ", ".join([f"{k.replace('_pct','').replace('_rate','').replace('pop_','p').replace('generations','g').replace('selection','s').replace('mutation','m')}={v}" for k, v in d.items()])
    def create_label(row):
        params = {k: row[k] for k in param_keys}
        if params == {
            'pop_size': 100, 
            'generations': 10, 
            'selection_pct': 0.4, 
            'mutation_rate': 0.2
        }:
            return "BASELINE (100, 10, 0.4, 0.2)"
        return dict_to_str(params)
    combo_labels = [dict_to_str(r) for r in results_df[param_keys].to_dict('records')]
    plt.figure(figsize=(15, 8))


    
    # Create the bar plot with error bars (std deviation)
    bars = plt.bar(
        combo_labels,
        results_df['effectiveness_avg_fitness'],
        yerr=results_df['effectiveness_std_dev'], # Add error bars
        capsize=5,
        color='skyblue',
        edgecolor='black'
    )
    for bar, label in zip(bars, combo_labels):
        if "BASELINE" in label:
            bar.set_color('salmon')
            bar.set_edgecolor('black')
    plt.ylabel('Effectiveness (Avg. Best Fitness)', fontsize=12)
    plt.xlabel('Parameter Combinations', fontsize=12)
    plt.title(f'GA Parameter Tuning Results (Avg. of {n_runs} runs each)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    plt.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    
    # Adjust ylim to make space for labels and error bars
    min_val = (results_df['effectiveness_avg_fitness'] - results_df['effectiveness_std_dev']).min()
    max_val = (results_df['effectiveness_avg_fitness'] + results_df['effectiveness_std_dev']).max()
    plt.ylim(bottom=min(0, min_val - 2), top=max_val + 5)
    
    plt.tight_layout()
    results_dir="output"
    filename = os.path.join(results_dir, "task5_tuning_results.png")
    plt.savefig(filename, dpi=300)
    plt.show()