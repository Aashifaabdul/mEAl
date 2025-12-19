# mutation_rate_experiment.py
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statistics import mean, stdev

# Import your project functions - adjust imports to your package layout
from src.task1_data_setup import load_and_prepare_data
from src.task2_algorithm_setup import add_nutrient_metrics, define_thresholds, fitness_method
from src.task5_parameter_tuning import * # Import the actual GA function

def run_custom_ga_once(df, thresholds, population_size, generations, cx_prob, mut_prob,
                       selection_method, truncation_pct, tournament_size, max_foods, seed):
    """
    Run a single GA experiment and return detailed results including history.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Run the genetic algorithm (adjust this call based on your actual function signature)
    try:
        # Try calling with your actual genetic_algorithm function
        result = run_ga_experiment(
            df=df,
            thresholds=thresholds,
            pop_size=population_size,
            generations=generations,
            mutation_rate=mut_prob,
            selection_pct=truncation_pct,
            max_foods=max_foods
        )
        
        # If result is just a fitness value, we need to track history ourselves
        if isinstance(result, (int, float)):
            # The function doesn't return history, so we'll need to modify our approach
            return {
                'best_fitness': result,
                'best_history': [result],  # Single point
                'avg_history': [result]
            }
        else:
            # If it returns a dict or tuple with history
            return result
            
    except Exception as e:
        print(f"Error in GA execution: {e}")
        return {
            'best_fitness': 0,
            'best_history': [0],
            'avg_history': [0]
        }

def run_mutation_sweep(output_dir='mutation_experiment',
                       mutation_rates=[0.05, 0.1, 0.2, 0.3, 0.4],
                       repeats=30,
                       fixed_params=None):
    os.makedirs(output_dir, exist_ok=True)
    df_raw, train_data, test_data = load_and_prepare_data()
    df = add_nutrient_metrics(df_raw)
    thresholds = define_thresholds(df)

    if fixed_params is None:
        fixed_params = {
            'population_size': 100,
            'generations': 40,
            'cx_prob': 0.8,
            'selection_method': 'truncation',
            'truncation_pct': 0.4,
            'tournament_size': 3,
            'max_foods': 4
        }

    rows = []
    convergence_histories = {mr: [] for mr in mutation_rates}

    run_id = 0
    for mr in mutation_rates:
        print(f"\nTesting mutation rate = {mr}")
        for r in range(repeats):
            run_id += 1
            seed = 1000 + run_id
            print(f"  Repeat {r+1}/{repeats}", end='\r')
            
            t0 = time.time()
            try:
                out = run_custom_ga_once(
                    df, thresholds,
                    population_size=fixed_params['population_size'],
                    generations=fixed_params['generations'],
                    cx_prob=fixed_params['cx_prob'],
                    mut_prob=mr,
                    selection_method=fixed_params['selection_method'],
                    truncation_pct=fixed_params['truncation_pct'],
                    tournament_size=fixed_params['tournament_size'],
                    max_foods=fixed_params['max_foods'],
                    seed=seed
                )
            except Exception as e:
                print(f"\nError in run {run_id}: {e}")
                continue
                
            t1 = time.time()
            runtime = t1 - t0
            
            # Safely extract results
            best_hist = out.get('best_history', [])
            avg_hist = out.get('avg_history', [])
            final_best = out.get('best_fitness', None)
            final_avg = avg_hist[-1] if avg_hist else None

            # convergence metric: first generation reaching >= 0.95 * final_best
            convergence_gen = None
            if final_best is not None and best_hist:
                threshold_val = 0.95 * final_best
                for i, v in enumerate(best_hist):
                    if v >= threshold_val:
                        convergence_gen = i + 1
                        break

            rows.append({
                'mutation_rate': mr,
                'repeat': r+1,
                'seed': seed,
                'runtime_s': runtime,
                'final_best': final_best,
                'final_avg': final_avg,
                'convergence_gen': convergence_gen,
                'best_history': best_hist
            })
            convergence_histories[mr].append(best_hist)
        
        print(f"  Completed {repeats} repeats")

    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'mutation_sweep_results.csv')
    
    # Save without the list column for CSV compatibility
    df_save = df_results.drop(columns=['best_history'])
    df_save.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    return df_results, convergence_histories

def plot_results(df_results, convergence_histories, output_dir='mutation_experiment'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out any failed runs (None values)
    df_plot = df_results[df_results['final_best'].notna()].copy()
    
    if df_plot.empty:
        print("No valid results to plot!")
        return
    
    # Boxplot of final_best by mutation_rate
    plt.figure(figsize=(10,6))
    mutation_rates = sorted(df_plot['mutation_rate'].unique())
    data_to_plot = [df_plot[df_plot['mutation_rate']==mr]['final_best'].values 
                    for mr in mutation_rates]
    
    plt.boxplot(data_to_plot, labels=[f'{mr:.2f}' for mr in mutation_rates])
    plt.xlabel('Mutation Rate', fontsize=12)
    plt.ylabel('Final Best Fitness', fontsize=12)
    plt.title('Distribution of Final Best Fitness by Mutation Rate', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_final_best.png'), dpi=200)
    plt.close()

    # Mean ± std bar chart
    stats = df_plot.groupby('mutation_rate')['final_best'].agg(['mean','std','count']).reset_index()
    
    plt.figure(figsize=(10,6))
    x_pos = np.arange(len(stats))
    plt.bar(x_pos, stats['mean'], yerr=stats['std'], capsize=5, alpha=0.7)
    plt.xticks(x_pos, [f'{mr:.2f}' for mr in stats['mutation_rate']])
    plt.xlabel('Mutation Rate', fontsize=12)
    plt.ylabel('Mean Final Best Fitness (± std)', fontsize=12)
    plt.title('Mean Final Best Fitness by Mutation Rate', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_mean_std_final_best.png'), dpi=200)
    plt.close()

    # Convergence curves: mean best per generation for each mutation rate
    plt.figure(figsize=(12,7))
    for mr, histories in convergence_histories.items():
        # Filter out empty histories
        valid_histories = [h for h in histories if h and len(h) > 0]
        if not valid_histories:
            continue
            
        # Pad histories to equal length with last value
        max_len = max(len(h) for h in valid_histories)
        padded = []
        for h in valid_histories:
            arr = list(h)
            if len(arr) < max_len:
                arr = arr + [arr[-1]]*(max_len-len(arr))
            padded.append(arr)
        
        data = np.array(padded, dtype=float)
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)
        
        generations = range(1, len(mean_curve)+1)
        plt.plot(generations, mean_curve, label=f'Mutation={mr:.2f}', linewidth=2)
        plt.fill_between(generations, 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Convergence Curves by Mutation Rate (Mean ± Std)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_curves.png'), dpi=200)
    plt.close()
    
    # Summary statistics table
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(stats.to_string(index=False))
    print("="*60)

if __name__ == '__main__':
    print("Starting Mutation Rate Experiment...")
    print(f"Testing rates: [0.05, 0.1, 0.2, 0.3, 0.4]")
    print(f"Repeats per rate: 30")
    print(f"Total runs: 150")
    print("-"*60)
    
    df_results, conv_hist = run_mutation_sweep(
        output_dir='mutation_experiment',
        mutation_rates=[0.05, 0.1, 0.2, 0.3, 0.4],
        repeats=30
    )
    
    plot_results(df_results, conv_hist, output_dir='mutation_experiment')
    print("\n✓ Experiment complete! Plots saved in 'mutation_experiment' folder.")