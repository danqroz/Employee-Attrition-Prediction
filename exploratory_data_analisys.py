import pandas as pd
import os


DATA_PATH = "data"
SAVE_PATH = os.path.join(DATA_PATH, "rais_national_2020-2021.parquet")

dfs = [
    pd.read_parquet(os.path.join(DATA_PATH, df)) for df in os.listdir(DATA_PATH)
    if df.endswith(".parquet")
]

df = pd.concat(dfs)
df["Regiao"] = df["Regiao"].apply(lambda x: x if x != "MG_ES_RJ" else "SUDESTE")
df.to_parquet(SAVE_PATH, engine="fastparquet", index=False)

df = pd.read_parquet(SAVE_PATH)
# for item in os.listdir(DATA_PATH):
#     if item != os.path.basename(SAVE_PATH):
#         os.remove(item)