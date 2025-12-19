# 🧬 MEAL – Meal Planning Evolutionary Algorithm

This project implements **mEAl**, an evolutionary algorithm designed to generate **nutritionally optimal meal plans** based on calorie limits and macro/micro-nutrient requirements.  

The project explores how different **computational intelligence techniques** perform on the same optimisation problem, including:
- A **custom Genetic Algorithm (mEAl)**
- **DEAP-based Genetic Algorithm**
- **Nature-inspired optimisation (Cuckoo Search)**
- Parameter tuning, diversity preservation, and bespoke genetic operators

---

## 📌 Project Objectives

The goal of this project is to:
- Design and implement an evolutionary algorithm that evolves **healthy meal combinations**
- Compare multiple optimisation strategies under identical constraints
- Analyse the impact of **selection, crossover, mutation, and diversity techniques**
- Evaluate algorithms based on **fitness, convergence, efficiency, and robustness**

A meal is considered healthy if it:
- Contains **500–800 kcal**
- Prioritises **high-quality macronutrients**
- Meets recommended **vitamin and mineral thresholds**
- Minimises unhealthy components (e.g. excess sodium, saturated fat)

---

## 📁 Project Structure

```text
src/
├── task1_data_setup.py              # Data loading, cleaning, preprocessing
├── task2_algorithm_setup.py         # Core GA design & fitness function
├── task3_DEAP.py                    # DEAP-based GA implementation
├── task4_nature_inspired_comparison.py  # Cuckoo Search
├── task5_parameter_tuning.py        # Systematic parameter experiments
├── task6_diversity_analysis.py      # Deterministic crowding
├── task7_bespoke_crossover.py       # Custom crossover operator
├── task11_fitness_sharing.py        # Fitness sharing diversity method

data/
├── food_dataset.csv
├── nutrition_dataset.csv

output/
├── *.png                            # Result plots & comparisons
├── *.csv                            # Experimental results

main.py                              # Entry point
README.md
```


---

## ⚙️ Setup Instructions

### 1️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
```

## Activate it:
### Windows
```bash
venv\Scripts\activate
```
### macOS / Linux
```basg
source venv/bin/activate
```
## 2️⃣Install dependencies
```bash
pip install numpy pandas scipy matplotlib deap openpyxl xlrd
```

## 3️⃣Run the project 
```bash
python main.py
```
## 📊 Outputs

The `output/` directory contains all experimental artefacts generated during execution of the algorithms.  
These outputs are used directly for **analysis, reporting, and presentation**.

### Contents

- **Fitness convergence plots**  
  Visualise how the best and average fitness values evolve across generations.

- **Algorithm comparison charts**  
  Compare the performance of mEAl, DEAP, and nature-inspired algorithms.

- **Parameter tuning results**  
  Results from systematic experiments evaluating the impact of population size, mutation rate, and generations.

- **Statistical analysis tables (`.csv`)**  
  Summarised metrics including effectiveness, efficiency, and success rate across multiple runs.



