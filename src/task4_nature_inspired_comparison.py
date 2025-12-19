"""
Nature-Inspired Optimization: Cuckoo Search for Meal Optimization
"""
import math
import numpy as np
import random
from src.task1_data_setup import load_and_prepare_data
from src.task2_algorithm_setup import *
from scipy import stats
import os 

# Algorithm parameters 
NUM_NESTS = 50
MAX_GENERATIONS = 10
PA = 0.25   # Discovery rate of alien eggs (probability to abandon a nest)
ALPHA = 0.01  # Step size for Lévy flight
MAX_FOODS = 4



def levy_flight(Lambda=1.5):
    """Generates a Lévy flight step size."""
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step

def random_meal(df):
    """Generates a random meal using the existing setup."""
    return create_random_population(df, size=1, max_foods=MAX_FOODS)[0]

def get_fitness(meal, df, thresholds):
    """Wrapper around the existing fitness function."""
    return fitness_method(meal, df, thresholds)

def generate_new_solution(meal, df):
    """Applies Lévy flight to modify quantities."""
    new_meal = []
    for food, qty in meal:
        new_qty = qty + ALPHA * levy_flight()
        new_qty = max(0.05, min(0.5, new_qty))  # constraint
        new_meal.append((food, round(new_qty, 2)))
    return new_meal

# Main Algorithm 
def run_cuckoo_search(df, thresholds, num_nests=NUM_NESTS, max_generations=MAX_GENERATIONS):
    """Run Cuckoo Search optimization."""
    nests = [random_meal(df) for _ in range(num_nests)]
    fitnesses = [get_fitness(meal, df, thresholds) for meal in nests]

    best_idx = np.argmax(fitnesses)
    best_nest = nests[best_idx]
    best_fitness = fitnesses[best_idx]

    best_hist, avg_hist = [], []
    for gen in range(max_generations):
        for i in range(num_nests):
            new_meal = generate_new_solution(nests[i], df)
            new_fitness = get_fitness(new_meal, df, thresholds)

            # Replace if better
            if new_fitness > fitnesses[i]:
                nests[i] = new_meal
                fitnesses[i] = new_fitness

        # Abandon a fraction of the worst nests
        num_abandon = int(PA * num_nests)
        worst_indices = np.argsort(fitnesses)[:num_abandon]
        for idx in worst_indices:
            nests[idx] = random_meal(df)
            fitnesses[idx] = get_fitness(nests[idx], df, thresholds)

        # Update best
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_nest = nests[current_best_idx]

        best_hist.append(best_fitness)
        avg_hist.append(np.mean(fitnesses))
    return best_nest, best_hist, avg_hist

def run_baseline_ga(df, thresholds, pop_size=100, generations=10, max_foods=4):
    """Baseline GA - returns best fitness"""
    population = create_random_population(df, size=pop_size, max_foods=max_foods)
    evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]
    
    best_fitness = max(fit for _, fit in evaluated) if evaluated else -float('inf')
    
    for gen in range(generations):
        evaluated.sort(key=lambda x: x[1], reverse=True)
        best_fitness = max(best_fitness, evaluated[0][1])
        
        selected = truncation_selection(evaluated, 0.4)
        if not selected:
            break
            
        parents = [m for m, _ in selected]
        new_gen = []
        
        num_children = pop_size - len(parents)
        for _ in range(num_children):
            if len(parents) < 2:
                parents.append(random.choice(evaluated)[0])
            child = set_based_crossover(random.choice(parents), random.choice(parents), max_foods=max_foods)
            child = mutate(child, mutation_rate=0.2)
            new_gen.append(child)
        
        next_population = parents + new_gen
        evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in next_population if meal]
    
    return best_fitness

def run_statistical_comparison(n_runs=30):
    """
    Simple statistical comparison with Welch's T-Test
    """
    
    print("\n" + "="*50)
    print("STATISTICAL COMPARISON")
    print("Baseline GA (mEAl) vs Cuckoo Search (NOM)")
    print("="*50)
    
    # Load data
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    # Hypotheses
    print("\nHYPOTHESES:")
    print("  H₀ (Null):        μ_baseline ≤ μ_cuckoo")
    print("  H₁ (Alternative): μ_baseline > μ_cuckoo")
    print("  α (Alpha):        0.05")
    
    # Run experiments
    print(f"\nRunning {n_runs} experiments for each algorithm...")
    
    baseline_results = []
    cuckoo_results = []
    
    for run in range(1, n_runs + 1):
        baseline_fitness = run_baseline_ga(df, thresholds)
        baseline_results.append(baseline_fitness)
        
        cuckoo_meal, _, _ = run_cuckoo_search(df, thresholds, num_nests=50, max_generations=10)
        cuckoo_fitness = fitness_method(cuckoo_meal, df, thresholds)
        cuckoo_results.append(cuckoo_fitness)
        
        if run % 10 == 0:
            print(f"  Completed {run}/{n_runs} runs")
    
    # Statistics
    baseline_mean = np.mean(baseline_results)
    cuckoo_mean = np.mean(cuckoo_results)
    
    print("\nRESULTS:")
    print(f"  Baseline GA Mean:   {baseline_mean:.2f}")
    print(f"  Cuckoo Search Mean: {cuckoo_mean:.2f}")
    print(f"  Difference:         {baseline_mean - cuckoo_mean:.2f}")
    
    # Welch's T-Test
    t_statistic, p_value_two_tailed = stats.ttest_ind(
        baseline_results, 
        cuckoo_results, 
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
        print(f"  REJECT H₀ (p = {p_value:.6f} < {alpha})")
        print(f"  Baseline GA (mEAl) significantly outperforms Cuckoo Search")
    else:
        print(f"   FAIL TO REJECT H₀ (p = {p_value:.6f} ≥ {alpha})")
        print(f"  No significant difference detected")
    
    print("="*70 + "\n")
    
    # Save results
    results_df = pd.DataFrame({
        'Run': range(1, n_runs + 1),
        'Baseline_GA': baseline_results,
        'Cuckoo_Search': cuckoo_results
    })
    results_dir="output"
    output_path = os.path.join(results_dir,'task4_comparison_results.csv')
    results_df.to_csv(output_path, index=False) 
    
    
    return {
        'baseline_mean': baseline_mean,
        'cuckoo_mean': cuckoo_mean,
        't_statistic': t_statistic,
        'p_value': p_value,
        'reject_null': p_value < alpha
    }




