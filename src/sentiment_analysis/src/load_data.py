from datasets import load_dataset
import pandas as pd

# Load IMDB dataset
dataset = load_dataset("imdb")

# Convert to DataFrame
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

# Save to CSV files
df_train.to_csv("../data/imdb_train.csv", index=False)
df_test.to_csv("../data/imdb_test.csv", index=False)

print("IMDB dataset downloaded and saved!")