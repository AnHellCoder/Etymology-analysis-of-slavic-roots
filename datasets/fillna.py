import pandas as pd

df = pd.read_csv('dataset_multi.csv')

df = df.fillna(-1)

df.to_csv('dataset_multi.csv')
