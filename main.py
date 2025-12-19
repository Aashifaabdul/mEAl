from src.task1_data_setup import *
from src.task2_algorithm_setup import *
from src.task3_DEAP import *
from src.task4_nature_inspired_comparison import *
from src.task5_parameter_tuning import *
from src.task6_diversity_analysis import *
from src.task7_bespoke_crossover import *
from src.task11_fitness_sharing import *
import random
import numpy as np
import matplotlib.pyplot as plt

def run_meal_evolution(max_reruns=5, min_acceptable_fitness=0):
    """
    Runs mEAl , tracks the generation of the best meal, 
    and retries if the minimum fitness threshold is not met.
    Returns: best_meal, fitness_history, avg_fitness_history
    """
    df, train_data, test_data = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)

    best_overall_meal = None
    best_overall_fitness = -float('inf')
    best_overall_generation = -1
    best_overall_run = -1
    
    # Track fitness history for visualization
    best_fitness_history = []
    avg_fitness_history = []
    
    reruns = 0
    
    while reruns < max_reruns:
        print(f"Running Meal Evolution (Attempt {reruns + 1} of {max_reruns})")
        
        population = create_random_population(df, size=100, max_foods=4) 
        evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]

        for generation in range(10):
            # Sort and select based on fitness (descending order)
            evaluated.sort(key=lambda x: x[1], reverse=True)
            
            # Track fitness for this generation
            fits = [fit for _, fit in evaluated]
            best_fitness_history.append(max(fits))
            avg_fitness_history.append(np.mean(fits))
            
            # Update best fitness for this run and overall
            current_best_meal, current_best_fitness = evaluated[0]

            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_meal = current_best_meal
                best_overall_generation = generation + 1
                best_overall_run = reruns + 1
            
            print(f"  Gen {generation+1:2d}: Best={max(fits):5.1f}, Avg={np.mean(fits):5.1f}, Min={min(fits):5.1f}")
            
            # Selection
            selected = truncation_selection(evaluated, 0.4)
            if not selected:
                population = create_random_population(df, size=100, max_foods=4)
                evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in population]
                continue
                
            parents = [m for m, _ in selected]
            new_gen = []

            # Crossover and Mutation
            num_children = len(population) - len(parents) 
            for _ in range(num_children):
                child = set_based_crossover(random.choice(parents), random.choice(parents), max_foods=4)
                child = mutate(child) 
                new_gen.append(child)

            # Combine parents and offspring and evaluate
            next_population = parents + new_gen
            evaluated = [(meal, fitness_method(meal, df, thresholds)) for meal in next_population if meal]

        
        # After 10 generations, check the best meal from this run
        evaluated.sort(key=lambda x: x[1], reverse=True)
        final_run_best_fitness = evaluated[0][1]
        
        # Check if the desired fitness is achieved
        if best_overall_fitness >= min_acceptable_fitness:
            print("\nSuccess! Optimal Meal Found.")
            break
        
        print(f" Run finished with max fitness: {final_run_best_fitness}. Retrying...")
        reruns += 1

    # Final Output
    if best_overall_fitness >= min_acceptable_fitness:
        print("\n Final Optimal Meal ")        
    print("--------------------------------------------------")
    print(f"Best Meal Found in Run {best_overall_run}, Generation {best_overall_generation}") 
    print(f"Fitness Score: {best_overall_fitness}")
    for food, qty in best_overall_meal:
        print(f"  {food}: {qty} kg")
    print("--------------------------------------------------")
    
    return best_overall_meal, best_fitness_history, avg_fitness_history


def run_deap_algorithm():
    """Run DEAP algorithm and track progress."""
    print("="*50)
    print("TASK 3 : RUNNING DEAP ALGORITHM")
    print("="*50)
    
    # Use the same data setup
    df, train_data, test_data = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    # Run DEAP from task3_DEAP.py (renamed import to deap_evolve)
    best_individual, best_hist, avg_hist = deap_evolve(df, thresholds, NGEN=10, CXPB=0.7, MUTPB=0.3)
    
    print("--------------------------------------------------")
    print(f"Best Fitness Score: {best_individual.fitness.values[0]:.1f}")
    print("\nBest Meal:")
    for food, qty in best_individual:
        print(f"  {food}: {qty} kg")
    print("--------------------------------------------------")
    print("\n")
    
    return best_individual, best_hist, avg_hist


def display_meal_comparison(custom_meal, deap_meal):
    """Display both meals side by side."""
    print("="*50)
    print("MEAL COMPARISON")
    print("="*50)
    
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    # Custom GA Meal
    print("mEAl - BEST MEAL:")
    print("-" * 50)
    custom_fitness = fitness_method(custom_meal, df, thresholds)
    print(f"Fitness: {custom_fitness:.1f}")
    print(f"Foods: {len(custom_meal)} items")
    
    from src.task2_algorithm_setup import compute_scaled_nutrients
    total_cal = sum(compute_scaled_nutrients(f, q, df)['Calories (kcal)'] for f, q in custom_meal)
    
    for i, (food, qty) in enumerate(custom_meal, 1):
        print(f"  {i}. {food:30s} : {qty:.2f} kg")
    print(f"Total Calories: {total_cal:.1f} kcal")
    
    # DEAP Meal
    print("-" * 50)
    print("\n DEAP - BEST MEAL:")
    print("-" * 50)
    
    if isinstance(deap_meal, creator.Individual):
        deap_meal_list = list(deap_meal)
    else:
        deap_meal_list = deap_meal
    
    deap_fitness = fitness_method(deap_meal_list, df, thresholds)
    print(f"Fitness: {deap_fitness:.1f}")
    print(f"Foods: {len(deap_meal_list)} items")
    
    total_cal_deap = sum(compute_scaled_nutrients(f, q, df)['Calories (kcal)'] for f, q in deap_meal_list)
    
    for i, (food, qty) in enumerate(deap_meal_list, 1):
        print(f"  {i}. {food:30s} : {qty:.2f} kg")
    print(f"Total Calories: {total_cal_deap:.1f} kcal")
    
    # Winner
    print("\n" + "-"*50)
    if custom_fitness > deap_fitness:
        print(f" WINNER: mEAl (by {custom_fitness - deap_fitness:.1f} points)")
    elif deap_fitness > custom_fitness:
        print(f" WINNER: DEAP (by {deap_fitness - custom_fitness:.1f} points)")
    else:
        print(" TIE: Both algorithms performed equally well!")
    


def visualize_comparison(custom_best_hist,deap_best_hist,cuckoo_best_hist=None):
    """Create comprehensive comparison visualizations."""
    
    # Load data for calculations
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    fig = plt.figure(figsize=(16, 10))
   # 1️ Best Fitness Convergence
    ax1 = plt.subplot(1, 2, 1)
    generations_custom = range(1, len(custom_best_hist) + 1)
    ax1.plot(generations_custom, custom_best_hist, 'b-o', label='Custom GA', linewidth=2.5, markersize=7)
    
    # Add DEAP curve if available
    if deap_best_hist is not None:
        generations_deap = range(1, len(deap_best_hist) + 1)
        ax1.plot(generations_deap, deap_best_hist, 'r-s', label='DEAP', linewidth=2.5, markersize=7)
    
    # Add Cuckoo curve if available
    if cuckoo_best_hist is not None:
        generations_cuckoo = range(1, len(cuckoo_best_hist) + 1)
        ax1.plot(generations_cuckoo, cuckoo_best_hist, 'g-^', label='Cuckoo Search', linewidth=2.5, markersize=7)
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Best Fitness Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2️ Final Best Fitness Comparison Bar Chart
    ax2 = plt.subplot(1, 2, 2)
    labels, fitness_vals, colors = ['Custom GA'], [custom_best_hist[-1]], ['#3498db']

    if deap_best_hist is not None:
        labels.append('DEAP')
        fitness_vals.append(deap_best_hist[-1])
        colors.append('#e74c3c')
        
    if cuckoo_best_hist is not None:
        labels.append('Cuckoo Search')
        fitness_vals.append(cuckoo_best_hist[-1])
        colors.append('#2ecc71')

    bars = ax2.bar(labels, fitness_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Final Best Fitness', fontsize=12, fontweight='bold')
    ax2.set_title('Final Best Fitness Comparison', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, max(fitness_vals) * 1.2])
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure dynamically
    if cuckoo_best_hist is not None:
        filename = os.path.join(results_dir, 'custom_deap_cuckoo_comparison.png') if deap_best_hist is not None else os.path.join(results_dir,'custom_cuckoo_comparison.png')
    else:
        filename = os.path.join(results_dir, "custom_deap_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
def run_cuckoo_algorithm():
    """Run Cuckoo Search and compare."""
    

    df, train_data, test_data = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)

    best_meal, best_hist, avg_hist = run_cuckoo_search(df, thresholds)
    print(f"Best Fitness Score: {fitness_method(best_meal, df, thresholds):.1f}")
    print("\nBest Meal:")
    for food, qty in best_meal:
        print(f"  {food}: {qty} kg")
    print("--------------------------------------------------")

    return best_meal, best_hist, avg_hist

if __name__ == "__main__":
    results_dir = "output"
    print("\n MEAL EVOLUTION ALGORITHM")
    print("="*50)
    
    # Run your existing Custom GA
    print(" TASK 2 : RUNNING mEAl")
    custom_meal, custom_best_hist, custom_avg_hist = run_meal_evolution(max_reruns=1, min_acceptable_fitness=0)
    
    # Run DEAP
    deap_meal, deap_best_hist, deap_avg_hist = run_deap_algorithm()
    
    # Display side-by-side comparison
    display_meal_comparison(custom_meal, deap_meal)
    
    # Comparison Summary
    df, _, _ = load_and_prepare_data()
    df = add_nutrient_metrics(df)
    thresholds = define_thresholds(df)
    
    custom_fitness = fitness_method(custom_meal, df, thresholds)
    deap_fitness = fitness_method(list(deap_meal) if isinstance(deap_meal, creator.Individual) else deap_meal, df, thresholds)

    
    # Create visualizations
    visualize_comparison(custom_best_hist, deap_best_hist)
    
    # Run Cuckoo Search
    print("="*50)
    print("\n TASK 4 : RUNNING CUCKOO SEARCH ALGORITHM...")
    print("="*50)
    cuckoo_meal, cuckoo_best_hist, cuckoo_avg_hist = run_cuckoo_algorithm()
    visualize_comparison(custom_best_hist,deap_best_hist,cuckoo_best_hist)
    results = run_statistical_comparison(n_runs=30)

    print("TASK 5: Parameter Management for mEAl...")

    PARAM_GRID = {
        'pop_size': [50, 150],             # Test small vs. large population
        'generations': [10, 30],           # Test short vs. long run
        'selection_pct': [0.2, 0.5],       # Test high vs. low selection pressure
        'mutation_rate': [0.1, 0.3]        # Test low vs. high mutation
    }
    
    # Number of times to run *each* combination to get a reliable average
    # run each 10 times.
    N_RUNS_PER_COMBINATION = 10
    
    # Define the fitness score required to consider a run "successful"
    SUCCESS_THRESHOLD_FITNESS = 15.0 

    # 3. Create all combinations
    params_keys = list(PARAM_GRID.keys())
    params_values = list(PARAM_GRID.values())
    all_combinations = list(product(*params_values))
    print("-"*50)
    print(f"Total parameter combinations to test: {len(all_combinations)}")
    print(f"Runs per combination: {N_RUNS_PER_COMBINATION}")
    print("-"*50)
    
    results = []
    experiment_start_time = time.time()

    # Run Baseline Algorithm First 
    print("\n Testing Baseline Configuration ")
    BASELINE_PARAMS = {
        'pop_size': 100,
        'generations': 10,
        'selection_pct': 0.4,
        'mutation_rate': 0.2,
        'max_foods': 4,
    }
    print(f"Configuration: {BASELINE_PARAMS}")
    print("-"*50)
    
    baseline_fitness_scores = []
    baseline_run_times = []
    for n in range(N_RUNS_PER_COMBINATION):
        run_start_time = time.time()
        score = run_ga_experiment(df, thresholds, **BASELINE_PARAMS)
        run_end_time = time.time()
        baseline_fitness_scores.append(score)
        baseline_run_times.append(run_end_time - run_start_time)

    # Calculate stats for baseline
    avg_fitness = np.mean(baseline_fitness_scores)
    std_fitness = np.std(baseline_fitness_scores)
    avg_run_time = np.mean(baseline_run_times)
    success_count = sum(1 for s in baseline_fitness_scores if s >= SUCCESS_THRESHOLD_FITNESS)
    success_rate_pct = (success_count / N_RUNS_PER_COMBINATION) * 100

    print(f"  => Stats: Avg Fitness={avg_fitness:.2f} | Success Rate={success_rate_pct:.0f}% | Avg Time={avg_run_time:.2f}s")
    
    # Store baseline result
    baseline_result_entry = BASELINE_PARAMS.copy()
    baseline_result_entry['success_rate_pct'] = success_rate_pct
    baseline_result_entry['effectiveness_avg_fitness'] = avg_fitness
    baseline_result_entry['effectiveness_std_dev'] = std_fitness
    baseline_result_entry['efficiency_avg_run_time_s'] = avg_run_time
    
    results.append(baseline_result_entry) # Add baseline to results list
    print("-"*50)
    print(f"Running Grid Search for {len(all_combinations)} Combinations")
    print(f"Total experiments to run: {len(all_combinations) * N_RUNS_PER_COMBINATION}\n")
    print("-"*50)

    # 4. Run the grid search
    for i, combo in enumerate(all_combinations):
        param_dict = dict(zip(params_keys, combo))
        
        # Check if this is the baseline combo, skip if it is
        if param_dict == BASELINE_PARAMS:
            print(f"Skipping Combo {i+1}/{len(all_combinations)} (Same as Baseline)")
            continue
            
        print(f" Testing Combo {i+1}/{len(all_combinations)}: {param_dict}")
        
        combo_fitness_scores = []
        combo_run_times = []

        for n in range(N_RUNS_PER_COMBINATION):
            run_start_time = time.time()
            # Run the lightweight experiment function
            score = run_ga_experiment(df, thresholds, **param_dict)
            run_end_time = time.time()
            
            combo_fitness_scores.append(score)
            combo_run_times.append(run_end_time - run_start_time)
            # print(f"  Run {n+1}/{N_RUNS_PER_COMBINATION}: Best Fitness = {score:.1f}")

        # 5. Calculate stats for this combination
        
        # Effectiveness (Solution Quality)
        avg_fitness = np.mean(combo_fitness_scores)
        std_fitness = np.std(combo_fitness_scores)
        
        # Efficiency (Speed)
        avg_run_time = np.mean(combo_run_times)
        
        # Success Rate
        success_count = sum(1 for s in combo_fitness_scores if s >= SUCCESS_THRESHOLD_FITNESS)
        success_rate_pct = (success_count / N_RUNS_PER_COMBINATION) * 100
        
        print(f"  => Stats: Avg Fitness={avg_fitness:.2f} | Success Rate={success_rate_pct:.0f}% | Avg Time={avg_run_time:.2f}s")
        
        # Store results with the requested terminology
        result_entry = param_dict.copy()
        result_entry['success_rate_pct'] = success_rate_pct
        result_entry['effectiveness_avg_fitness'] = avg_fitness
        result_entry['effectiveness_std_dev'] = std_fitness
        result_entry['efficiency_avg_run_time_s'] = avg_run_time
        results.append(result_entry)

    print("\n")
    print("EXPERIMENT COMPLETE")
    print(f"Total duration: {(time.time() - experiment_start_time) / 60:.2f} minutes")
    print("-"*50)

    # 6. Report findings
    if not results:
        print("No results to report.")
    else:
        results_df = pd.DataFrame(results)
        
        # Sort by best effectiveness, then best success rate
        results_df = results_df.sort_values(
            by=['effectiveness_avg_fitness', 'success_rate_pct'], 
            ascending=[False, False]
        )
        output_path = os.path.join(results_dir, "task5_tuning_results.csv")
        results_df.to_csv(output_path, index=False)   
         # 1. Create the 'Parameter Configurations' column
        param_cols = list(PARAM_GRID.keys())
        # Convert parameter columns to a dictionary string for the first column
        
        def create_param_label(row):
            params = {k: row[k] for k in param_cols}
            if params == BASELINE_PARAMS:
                return f"BASELINE: {params}"
            return str(params)
            
        results_df['param_config_dict'] = results_df.apply(create_param_label, axis=1)
        
        # 2. Create the 'Performance Measures' column
        # Helper function to create the metrics dictionary for the second column
        def create_metrics_dict(row):
            return {
                'success_rate': f"{row['success_rate_pct']:.0f}%",
                'effectiveness': f"{row['effectiveness_avg_fitness']:.2f} (Â±{row['effectiveness_std_dev']:.2f})",
                'efficiency': f"{row['efficiency_avg_run_time_s']:.2f}s"
            }

        results_df['performance_measures'] = results_df.apply(create_metrics_dict, axis=1)
        
        # 3. Create the final 2-column DataFrame
        final_table_df = results_df[['param_config_dict', 'performance_measures']]
        
        # Rename columns for the final print
        final_table_df.columns = ["Parameter Configurations", "Performance Measures"]
        
        # Print the 2-column table
        # We use pandas' max_colwidth to prevent the dictionaries from being truncated
        with pd.option_context('display.max_colwidth', None):
            print(final_table_df.to_string(index=False))

    
        # Original Best Combination Report
        print("\n")
        print(" BEST PARAMETER COMBINATION ")
        print("Based on highest Effectiveness (avg fitness) and Success Rate:")
        # We need to present the best one clearly from the results_df
        best_combo_details = results_df.iloc[0]
        print(f"Configuration: {best_combo_details['param_config_dict']}")
        print(f"Performance: {best_combo_details['performance_measures']}")
        print("="*70)
        
        # 7. Visualize the primary metric (Effectiveness)
        # Note: plot_results still needs the original param_keys and the full results_df
        plot_results(results_df, params_keys, N_RUNS_PER_COMBINATION)


    print("TASK 6: POPULATION DIVERSITY ANALYSIS (DETERMINISTIC CROWDING)")
    N_RUNS_PER_ALGORITHM = 10 
    SUCCESS_THRESHOLD_FITNESS = 15.0 

    # 2. Define all algorithms to test
    ALGORITHMS_TO_TEST = [
        ("Baseline GA (mEAl)", run_ga_baseline),
        ("Deterministic Crowding", run_ga_crowding),
    ]    
    all_results = []
    experiment_start_time = time.time()
    BASELINE_PARAMS = {
    'pop_size': 100,
    'generations': 10,
    'max_foods': 4,
    'mutation_rate': 0.2,
    #Parameters specific to techniques 
    'truncation_pct': 0.4,   # For Baseline
    'niche_radius': 0.5,     # For Fitness Sharing (Jaccard distance)
}

    # 3. Run the experiment for each algorithm
    for name, algorithm_func in ALGORITHMS_TO_TEST:
        
        print(f"Testing: {name} ")
        
        algo_fitness_scores = []
        algo_run_times = []

        for n in range(N_RUNS_PER_ALGORITHM):
            run_start_time = time.time()
            score = algorithm_func(df, thresholds, BASELINE_PARAMS)
            run_end_time = time.time()
            
            algo_fitness_scores.append(score)
            algo_run_times.append(run_end_time - run_start_time)

        # 4. Calculate stats for this algorithm
        
        # Effectiveness (Solution Quality)
        avg_fitness = np.mean(algo_fitness_scores)
        std_fitness = np.std(algo_fitness_scores)
        
        # Efficiency (Speed)
        avg_run_time = np.mean(algo_run_times)
        
        # Success Rate
        success_count = sum(1 for s in algo_fitness_scores if s >= SUCCESS_THRESHOLD_FITNESS)
        success_rate_pct = (success_count / N_RUNS_PER_ALGORITHM) * 100
        
        print(f"  => Stats: Effectiveness={avg_fitness:.2f} | Success Rate={success_rate_pct:.0f}% | Efficiency={avg_run_time:.2f}s")
    
    task6_results = run_statistical_comparison_diversity(n_runs=30)

    print("TASK 7 : Enhancing the genetic operator (Cross-over operator)...")
    task7results=run_statistical_comparison_crossover(n_runs=30)

    print("TASK 11: Population diversity analysis (Fitness sharing)")
    fit_shar=run_statistical_comparison_diversity_fitshar(n_runs=30)
        