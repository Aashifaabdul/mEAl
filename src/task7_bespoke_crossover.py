"""
Enhanced mEAl Algorithm with Bespoke Crossover Operator

Nutrient-Aware Fitness-Weighted Crossover
- Instead of random selection, prioritizes foods with better nutrient profiles
- Uses fitness evaluation to guide food selection from parents
- Maintains diversity while exploiting known good solutions
"""

import numpy as np
import pandas as pd
from scipy import stats
import random
import os
from src.task1_data_setup import load_and_prepare_data
from src.task2_algorithm_setup import *

results_dir="output"
def bespoke_nutrient_aware_crossover(p1, p2, df, thresholds, max_foods=4):
    """
    Nutrient-Aware Fitness-Weighted Crossover
    This exploits domain knowledge about nutrition while maintaining
    genetic diversity through probabilistic selection.
    Args:
        p1, p2: Parent meals [(food, qty), ...]
        df: DataFrame with nutrient data
        thresholds: Nutrient thresholds
        max_foods: Maximum foods in offspring
        
    Returns:
        child_meal: New meal [(food, qty), ...]
    """
    
    # Convert parents to dictionaries
    d1 = {f: q for f, q in p1}
    d2 = {f: q for f, q in p2}
    all_foods = list(set(d1) | set(d2))
    
    # Handle edge case: no foods
    if not all_foods:
        return [(random.choice(list(df.index)), 0.2)]
    
    # Evaluate individual food fitness contributions
    food_scores = {}
    for food in all_foods:
        # Get average quantity from parents
        avg_qty = (d1.get(food, 0) + d2.get(food, 0)) / 2
        if avg_qty == 0:
            avg_qty = d1.get(food, d2.get(food, 0.2))
        
        # Create single-food meal and evaluate
        single_meal = [(food, avg_qty)]
        score = fitness_method(single_meal, df, thresholds)
        food_scores[food] = max(score, 0.1)  # Avoid negative scores
    
    # Convert scores to probabilities (softmax-like)
    total_score = sum(food_scores.values())
    food_probs = {f: s / total_score for f, s in food_scores.items()}
    
    # Select foods based on fitness-weighted probabilities
    foods_list = list(food_probs.keys())
    probs_list = [food_probs[f] for f in foods_list]
    
    # Ensure valid range for randint
    min_foods = min(2, len(foods_list))
    max_foods_available = min(max_foods, len(foods_list))
    num_foods = random.randint(min_foods, max(min_foods, max_foods_available))
    
    # Weighted selection without replacement
    selected_foods = []
    remaining_foods = foods_list.copy()
    remaining_probs = probs_list.copy()
    
    for _ in range(num_foods):
        if not remaining_foods:
            break
        
        # Normalize probabilities
        prob_sum = sum(remaining_probs)
        if prob_sum == 0:
            break
        norm_probs = [p / prob_sum for p in remaining_probs]
        
        # Select food
        chosen_idx = np.random.choice(len(remaining_foods), p=norm_probs)
        selected_foods.append(remaining_foods[chosen_idx])
        
        # Remove selected food
        remaining_foods.pop(chosen_idx)
        remaining_probs.pop(chosen_idx)
    
    # Determine quantities using fitness-weighted averaging
    child_meal = []
    for food in selected_foods:
        q1 = d1.get(food, 0)
        q2 = d2.get(food, 0)
        
        if q1 > 0 and q2 > 0:
            # Both parents have this food - weighted average
            # Weight by parent fitness
            parent1_fit = fitness_method(p1, df, thresholds)
            parent2_fit = fitness_method(p2, df, thresholds)
            
            # Avoid negative fitness causing issues
            w1 = max(parent1_fit, 0.1)
            w2 = max(parent2_fit, 0.1)
            
            qty = (q1 * w1 + q2 * w2) / (w1 + w2)
        elif q1 > 0:
            qty = q1
        else:
            qty = q2
        
        # Add small random variation
        qty = qty * random.uniform(0.9, 1.1)
        qty = round(max(0.05, min(qty, 0.5)), 2)
        
        child_meal.append((food, qty))
    
    return child_meal


# ORIGINAL CROSSOVER (for comparison)
def original_crossover(p1, p2, max_foods=4):
    """Original set-based crossover from task2"""
    d1 = {f: q for f, q in p1}
    d2 = {f: q for f, q in p2}
    all_foods = set(d1) | set(d2)

    combined_quantities = {
        f: round((d1.get(f, 0) + d2.get(f, 0)) / 2, 2) if f in d1 and f in d2 
        else d1.get(f, d2.get(f)) 
        for f in all_foods
    }

    combined_list = list(combined_quantities.items())
    num_foods_to_select = random.randint(1, min(max_foods, len(combined_list)))
    child_meal = random.sample(combined_list, num_foods_to_select)

    return child_meal

# GA IMPLEMENTATIONS

def run_ga_with_crossover(df, thresholds, crossover_func, crossover_name, 
                          pop_size=100, generations=10, max_foods=4):
    """
    Run GA with specified crossover operator
    
    Returns:
        best_fitness: Best fitness achieved
        fitness_history: Best fitness per generation
    """
    population = create_random_population(df, size=pop_size, max_foods=max_foods)
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]
    
    best_fitness = max(fit for _, fit in evaluated) if evaluated else -float('inf')
    fitness_history = []
    
    for gen in range(generations):
        evaluated.sort(key=lambda x: x[1], reverse=True)
        current_best = evaluated[0][1]
        best_fitness = max(best_fitness, current_best)
        fitness_history.append(best_fitness)
        
        # Selection
        selected = truncation_selection(evaluated, 0.4)
        if not selected:
            break
            
        parents = [m for m, _ in selected]
        new_gen = []
        
        # Crossover and Mutation
        num_children = pop_size - len(parents)
        for _ in range(num_children):
            if len(parents) < 2:
                parents.append(random.choice(evaluated)[0])
            
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            # Apply the specified crossover
            if crossover_name == "Bespoke":
                child = crossover_func(p1, p2, df, thresholds, max_foods=max_foods)
            else:
                child = crossover_func(p1, p2, max_foods=max_foods)
            
            child = mutate(child, mutation_rate=0.2)
            new_gen.append(child)
        
        next_population = parents + new_gen
        evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in next_population if meal]
    
    return best_fitness, fitness_history


# STATISTICAL COMPARISON

def run_statistical_comparison_crossover(n_runs=30):
    """
    Compare Original vs Bespoke Crossover with statistical testing
    """
    print("-"*50)
    print("Original mEAl vs Enhanced mEAl (Nutrient-Aware Crossover)")
    print("="*50)
    
    # Load data
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    # Hypotheses
    print("\nHYPOTHESES:")
    print("  H₀ (Null):        μ_bespoke ≤ μ_original")
    print("  H₁ (Alternative): μ_bespoke > μ_original")
    print("  α (Alpha):        0.05")
    
    # Run experiments
    print(f"\nRunning {n_runs} experiments for each version...")
    
    original_results = []
    bespoke_results = []
    
    original_generations = []
    bespoke_generations = []
    
    for run in range(1, n_runs + 1):
        # Original crossover
        fitness_orig, hist_orig = run_ga_with_crossover(
            df, thresholds, original_crossover, "Original"
        )
        original_results.append(fitness_orig)
        original_generations.append(hist_orig)
        
        # Bespoke crossover
        fitness_besp, hist_besp = run_ga_with_crossover(
            df, thresholds, bespoke_nutrient_aware_crossover, "Bespoke"
        )
        bespoke_results.append(fitness_besp)
        bespoke_generations.append(hist_besp)
        
        if run % 10 == 0:
            print(f"  Completed {run}/{n_runs} runs")
    
    # Statistics
    original_mean = np.mean(original_results)
    bespoke_mean = np.mean(bespoke_results)
    
    print("\nRESULTS:")
    print(f"  Original mEAl Mean:  {original_mean:.2f}")
    print(f"  Bespoke mEAl Mean:   {bespoke_mean:.2f}")
    print(f"  Improvement:         {bespoke_mean - original_mean:.2f} ({((bespoke_mean - original_mean) / original_mean * 100):.1f}%)")
    
    # Computational efficiency: convergence speed
    # Find average generation to reach 90% of final fitness
    def convergence_gen(history, target_pct=0.9):
        if not history or len(history) == 0:
            return len(history)
        final_fitness = history[-1]
        target = final_fitness * target_pct
        for i, fit in enumerate(history):
            if fit >= target:
                return i + 1
        return len(history)
    
    orig_conv = np.mean([convergence_gen(h) for h in original_generations])
    besp_conv = np.mean([convergence_gen(h) for h in bespoke_generations])
    
    print("\nCOMPUTATIONAL EFFICIENCY:")
    print(f"  Original: Avg generations to 90% fitness: {orig_conv:.1f}")
    print(f"  Bespoke:  Avg generations to 90% fitness: {besp_conv:.1f}")
    print(f"  Speedup:  {((orig_conv - besp_conv) / orig_conv * 100):.1f}% faster")
    
    # Welch's T-Test
    t_statistic, p_value_two_tailed = stats.ttest_ind(
        bespoke_results, 
        original_results, 
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
        print(f"  Bespoke crossover significantly outperforms original")
        print(f"  The nutrient-aware approach provides measurable improvement!!")
    else:
        print(f"   FAIL TO REJECT H₀ (p = {p_value:.6f} ≥ {alpha})")
        print(f"  No significant difference detected")
    
    print("="*50 + "\n")
    
    # Save results
    results_df = pd.DataFrame({
        'Run': range(1, n_runs + 1),
        'Original_mEAl': original_results,
        'Bespoke_mEAl': bespoke_results,
        'Improvement': np.array(bespoke_results) - np.array(original_results)
    })

    output_path = os.path.join(results_dir,'task7_comparison_results.csv')
    results_df.to_csv(output_path, index=False)

    return {
        'original_mean': original_mean,
        'bespoke_mean': bespoke_mean,
        't_statistic': t_statistic,
        'p_value': p_value,
        'reject_null': p_value < alpha,
        'improvement_pct': ((bespoke_mean - original_mean) / original_mean * 100),
        'convergence_speedup': ((orig_conv - besp_conv) / orig_conv * 100)
    }
