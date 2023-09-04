from ftplib import FTP
import io
import random
import pandas as pd
import os
import typer
import tempfile
import py7zr
import numpy as np


ftp_host = 'ftp.mtps.gov.br'
rais_path = '/pdet/microdados/RAIS/'
SAVE_PATH = "data"

to_drop = [
    "Bairros SP",
    "Bairros Fortaleza",
    "Bairros RJ",
    "CNAE 95 Classe",
    "Distritos SP",
    "Faixa Etária",
    "Faixa Hora Contrat",
    "Ind CEI Vinculado",
    "Ind Simples",
    "Mun Trab",
    "Nacionalidade",
    "Regiões Adm DF",
    "Tipo Estab.1",
    "Vl Remun Dezembro Nom",
    "Vl Remun Média Nom",
    "Vl Remun Dezembro (SM)",
    "Vl Rem Janeiro SC",
    "Vl Rem Fevereiro SC",
    "Vl Rem Março SC",
    "Vl Rem Abril SC",
    "Vl Rem Maio SC",
    "Vl Rem Junho SC",
    "Vl Rem Julho SC",
    "Vl Rem Agosto SC",
    "Vl Rem Setembro SC",
    "Vl Rem Outubro SC",
    "Vl Rem Novembro SC",
    "Ano Chegada Brasil",
]


def _get_total_lines(text_path):
    with open(text_path, "r", encoding='latin1') as file:
        total_lines = sum(1 for line in file)
    return total_lines


def _random_lines(n_lines, sample_proportion=.01, seed=10):
    random.seed(seed)
    sample_size = int(n_lines * sample_proportion)
    sample_idxs = random.sample(range(1, n_lines), sample_size)
    return np.array([0] + sorted(sample_idxs))


def _extract_txt_from_7z(archive, txt_file_name, low_memory):
    with py7zr.SevenZipFile(archive, mode='r') as z:
        with tempfile.TemporaryDirectory() as temp_dir:
            z.extract(targets=[txt_file_name], path=temp_dir)
            txt_path = os.path.join(temp_dir, txt_file_name)
            print("Sampling file...")
            if low_memory:
                sample_lines = list()
                total_lines = _get_total_lines(txt_path)
                sample_idxs = _random_lines(total_lines)
                with open(txt_path, 'r', encoding='latin1') as txt_file:
                    for idx, line in enumerate(txt_file):
                        if idx in sample_idxs:
                            sample_lines.append(line)
            else:
                with open(txt_path, 'r', encoding='latin1') as txt_file:
                    content_lines = txt_file.readlines()
                    sample_idxs = _random_lines(len(content_lines))
                    sample_lines = [content_lines[i] for i in sample_idxs]
    return sample_lines


def get_content_locally(file_name, ftp, low_memory):
    with io.BytesIO() as local_file:
        ftp.retrbinary('RETR ' + file_name, local_file.write)
        local_file.seek(0)
        txt = file_name.replace("7z", "txt")
        print("Extracting file: ", txt)
        sample = _extract_txt_from_7z(local_file, txt, low_memory)
    return sample


def file_exists(name):
    full_path = os.path.join(SAVE_PATH, name)
    return os.path.isfile(full_path)


def create_sample_dataframe(data):
    file_like_content = io.StringIO(data)
    df = pd.read_csv(
        file_like_content,
        sep=";",
        decimal=",",
        header=0,
        usecols=lambda col: col not in to_drop,
        dtype={
            "CBO Ocupação 2002": str,
            "Mês Desligamento": str
        }
    )
    return df


def get_region(file_name):
    return file_name.rsplit(".", 1)[0].split("PUB_")[1]


def save_data(df, name):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print("Saving data on: ", SAVE_PATH)
    # df.to_csv(os.path.join(SAVE_PATH, name), index=False, sep=";")
    df.to_parquet(os.path.join(SAVE_PATH, name), engine="fastparquet")


def save_dataframe(year, folder, ftp, low_memory=False):
    name = f'{year}_{folder.replace("7z", "parquet")}'
    if file_exists(name):
        print(f"The file {name} already exists.")
        return None
    print("Acessing dir: ", folder)
    sample = get_content_locally(folder, ftp, low_memory)
    region = get_region(folder)

    print("Creating dataframe for: ", region)
    df = create_sample_dataframe("\r\n".join(sample))
    df["Regiao"] = region
    df["Ano"] = year
    save_data(df, name)
    print("Done!")
    print()


def main(year):
    path = f"{rais_path}{year}"
    ftp = FTP(ftp_host)
    ftp.login()
    ftp.cwd(path)

    folders = [file for file in ftp.nlst() if file != "RAIS_ESTAB_PUB.7z"]

    for folder in folders:
        save_dataframe(year, folder, ftp)


if __name__ == "__main__":
    typer.run(main)
