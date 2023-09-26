import random
import os
import pandas as pd

TRAIN_PATH = os.path.join("data", "train_rais_2020.parquet")
OOT_PATH = os.path.join("data", "oot_rais_2021.parquet")

random.seed(42)
target_window = 6


def _get_random_window(hiring_month, resig_month):
    return random.randint(hiring_month + 1, min(target_window, resig_month))


def _create_target(df):
    return (df["mes_desligamento"] <= df["month_ref"] + target_window).astype(int)


def _update_length_employment(df):
    return df["tempo_emprego"] - df["mes_desligamento"] + df["month_ref"]


def _get_target_window(hiring_month, resig_month):
    return range(hiring_month + 1, min(resig_month, target_window) + 1)


def _save_dataframe(df, path):
    df.to_parquet(path, engine="fastparquet", index=False)


def main():
    df_train = pd.read_parquet(TRAIN_PATH)
    df_oot = pd.read_parquet(OOT_PATH)

    df_train["month_ref"] = df_train.apply(
        lambda row: _get_random_window(row["mes_admissao"], row["mes_desligamento"]),
        axis=1
    )

    df_train["tempo_emprego"] = _update_length_employment(df_train)
    df_train["target"] = _create_target(df_train)

    df_oot["month_ref"] = df_oot.apply(
        lambda row: _get_target_window(row["mes_admissao"], row["mes_desligamento"]),
        axis=1
    )
    df_oot = df_oot.explode("month_ref")
    df_oot["tempo_emprego"] = _update_length_employment(df_oot)
    df_oot["target"] = _create_target(df_oot)
    # DROPAR MES DESLIGAMENTO, MOTIVO DESLIGAMENTO, MONTH_REF
    _save_dataframe(df_train, TRAIN_PATH)
    _save_dataframe(df_oot, OOT_PATH)


if __name__ == "__main__":
    main()
