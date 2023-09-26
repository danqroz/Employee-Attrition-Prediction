import pandas as pd
import os
from unidecode import unidecode


target_window = 6


def _drop_deceased_retired(df):
    col = "motivo_desligamento"
    return df.query(f"not (60 <= {col} <= 80)")


def _drop_temp_employment(df):
    return df.query("tipo_vinculo not in (50, 90, 95)")


def _drop_provisional_admission(df):
    return df.query("tipo_admissao != 10")


def _rename_region(df):
    df["regiao"] = df["regiao"].apply(lambda x: x if x != "MG_ES_RJ" else "SUDESTE")
    return df


def _drop_recent_employers(df):
    return df.query("tempo_emprego >= 1")


def _idade_range(df):
    return df.query("14 <= idade <= 65")


def _drop_null_salary(df):
    return df.query("`vl_remun_media_(sm)` > 0")


def _drop_occupation_nan(df):
    return df.query("cbo_ocupacao_2002 != '0000-1'")


def _drop_null_lengh_employment(df):
    return df.query("tempo_emprego >= 1")


def _drop_cols(df):
    return df.drop(columns=[
        'vinculo_ativo_31/12',
        'faixa_remun_dezem_(sm)',
        'faixa_tempo_emprego',
        'faixa_remun_media_(sm)',
        'ind_portador_defic',
        'tipo_defic',
        'raca_cor',
    ])


def _rename_cols(df):
    df.columns = df.columns.map(
        lambda x: unidecode(x).replace(" ", "_").lower() if isinstance(x, (str)) else x
    )
    return df


def _concat_all_data():
    dfs = [
        pd.read_parquet(os.path.join("data", df)) for df in os.listdir("data")
        if df.startswith("_RAIS_VINC_PUB_", 4) and df.endswith(".parquet")
    ]

    df = pd.concat(dfs)
    return df


def _save_dataframe(df, path):
    df.to_parquet(path, engine="fastparquet", index=False)


def clean_df(df):
    df = df.pipe(_rename_cols).pipe(_drop_cols).pipe(_rename_region).pipe(
        _drop_deceased_retired)

    df["mes_desligamento"].replace({"{ñ": 13}, inplace=True)
    df["mes_desligamento"] = df["mes_desligamento"].astype(int)

    df = df.query("mes_admissao < @target_window")
    df = df.query("mes_admissao < mes_desligamento")
    return df


def main():
    df = _concat_all_data()
    df = clean_df(df)

    train_df = df[df["ano"] == "2020"]
    df_oot = df[df["ano"] == "2021"]
    train_df = train_df.pipe(_drop_temp_employment).pipe(_idade_range).pipe(
        _drop_provisional_admission).pipe(_drop_recent_employers).pipe(
        _drop_null_salary).pipe(_drop_null_lengh_employment).pipe(_drop_occupation_nan)

    _save_dataframe(train_df, os.path.join("data", "train_rais_2020.parquet"))
    _save_dataframe(df_oot, os.path.join("data", "oot_rais_2021.parquet"))


if __name__ == "__main__":
    main()
