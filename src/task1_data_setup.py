import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path="data/food_dataset.csv"):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    df = data.set_index('Nutrient').T
    df = df.replace('trace', 0.1) #replace trace values 
    #print(df)
    # Convert all numeric columns properly
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    #print(train_data)
    return df, train_data, test_data
    
