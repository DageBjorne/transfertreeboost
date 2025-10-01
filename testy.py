import pandas as pd
import io

# Läs in din CSV-data
df = pd.read_csv('results/tradaboost_ablation_rs_norrland.csv')

# Läs in datan i en DataFrame
#df = pd.read_csv(io.StringIO(csv_data))

# Gruppera datan och beräkna medelvärdet av val_rmse
grouped_data = df.groupby([
    'target_column', 'train_size_', 'method', 'n_estimators', 'lr', 'tree_size'
])['val_rmse'].mean()

grouped_data.to_csv('testy.csv')
