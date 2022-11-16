import pandas as pd


def data_downloader(csv_path:str):
    data = pd.read_csv(csv_path)

    print(data)