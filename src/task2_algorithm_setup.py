import pandas as pd
import random
#Nutrients metrics 
def getwtrsolvits(row):
    return row.get('Vitamin B (mg)', 0) + row.get('Vitamin C (mg)', 0)

def getfatsolvits(row):
    return row.get('Vitamin A (µg)', 0)

def getgoodminr(row):
    minerals = ['Iron (mg)', 'Magnesium (mg)', 'Phosphorus (mg)', 
                'Potassium (mg)', 'Zinc (mg)', 'Manganese (mg)', 'Selenium (Âµg)']
    return sum(row.get(m, 0) for m in minerals)

def getpotbadminr(row):
    return row.get('Sodium (mg)', 0)

def add_nutrient_metrics(df):
    df['wtrsolvits'] = df.apply(getwtrsolvits, axis=1)
    df['fatsolvits'] = df.apply(getfatsolvits, axis=1)
    df['goodminr'] = df.apply(getgoodminr, axis=1)
    df['potbadminr'] = df.apply(getpotbadminr, axis=1)
    return df

def define_thresholds(df):
    return {
        'MIN_WTR_SOL_VIT': df['wtrsolvits'].quantile(0.25),
        'MIN_FAT_SOL_VIT': df['fatsolvits'].quantile(0.25),
        'MAX_FAT_SOL_VIT': df['fatsolvits'].quantile(0.75),
        'MIN_GOOD_MINERAL': df['goodminr'].quantile(0.25),
        'MIN_BAD_MINERAL': df['potbadminr'].quantile(0.25),
        'MAX_BAD_MINERAL': df['potbadminr'].quantile(0.75)
    }

def get_food_series(food_name, df):
    return df.loc[food_name]

def compute_scaled_nutrients(food_name, qty, df):
    factor = qty * 10  # nutrients per 100g
    s = get_food_series(food_name, df)
    scaled = s.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            scaled[col] *= factor
    return scaled

def fitness_method(meal, df, thresholds):# fitness value checked 
    fitval, total_calories = 0, 0
    for food, qty in meal:
        nutrients = compute_scaled_nutrients(food, qty, df)
        total_calories += nutrients['Calories (kcal)']
        fitval += sum([
            1 if nutrients['wtrsolvits'] >= thresholds['MIN_WTR_SOL_VIT'] else -1,
            1 if thresholds['MIN_FAT_SOL_VIT'] <= nutrients['fatsolvits'] <= thresholds['MAX_FAT_SOL_VIT'] else -1,
            1 if nutrients['goodminr'] >= thresholds['MIN_GOOD_MINERAL'] else -1,
            1 if thresholds['MIN_BAD_MINERAL'] <= nutrients['potbadminr'] <= thresholds['MAX_BAD_MINERAL'] else -1
        ])
    fitval += 5 if 500 <= total_calories <= 800 else -10 # +5 for calorie between 500-800 else -10
    return fitval

def create_random_population(df, size=50, max_foods=5, min_qty=0.05, max_qty=0.5):
    foods = list(df.index)
    pop = []
    for _ in range(size):
        num_foods = random.randint(1, max_foods)
        selected = random.sample(foods, num_foods)
        meal = [(food, round(random.uniform(min_qty, max_qty), 2)) for food in selected]
        pop.append(meal)
    return pop

def truncation_selection(population, pct=0.5): #truncation selection 
    population.sort(key=lambda x: x[1], reverse=True)
    return population[:int(len(population) * pct)]

def set_based_crossover(p1, p2, max_foods=4): # set_based_crossover
    d1 = {f: q for f, q in p1}
    d2 = {f: q for f, q in p2}
    all_foods = set(d1) | set(d2)

    # 1. Calculate the averaged quantity for all possible foods
    combined_quantities = {
        f: round((d1.get(f, 0) + d2.get(f, 0)) / 2, 2) if f in d1 and f in d2 
        else d1.get(f, d2.get(f)) 
        for f in all_foods
    }

    # 2. Convert to a list of (food, quantity) tuples
    combined_list = list(combined_quantities.items())
    
    # 3. Select: Randomly sample up to max_foods items 
    #    from the combined list to form the child meal
    num_foods_to_select = random.randint(1, min(max_foods, len(combined_list)))
    
    # Randomly select the food items for the child meal
    child_meal = random.sample(combined_list, num_foods_to_select)
    return child_meal

def mutate(meal, mutation_rate=0.2):
    mutated = []
    for food, qty in meal:
        if random.random() < mutation_rate:
            qty = round(qty + random.uniform(-0.05, 0.05), 2)
            qty = max(0.05, min(qty, 0.5))
        mutated.append((food, qty))
    return mutated
