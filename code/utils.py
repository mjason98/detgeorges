from xmlrpc.client import boolean
import pandas as pd
import requests
import os
from tqdm import tqdm

def add_color(text):
	return '\033[91m' + text + '\033[0m'

def data_images_downloader(csv_path:str, image_folder:str ="images", data_folder:str ="data", force_download=False):
    '''
    Download images from a CSV file that provides the URLs to the images.

    Parameters:
        csv_path: str   A path to a csv file, where the data of the first column are URLs to images.

        data_folder: str    A path to the directory where the folder to save the images will be created: <data_folder>/<image_folder> 
        
        image_folder: str    The folder's name where the images will be saved.

        force_download:     Download the image even if it is already present in the path to be saved.
    '''

    image_folder = os.path.join(data_folder, image_folder)

    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    data = pd.read_csv(csv_path, header=None)

    print ("# Downloading data from", add_color(csv_path))

    iter = tqdm(range(len(data)))
    for i in iter:
        url = str(data.iloc[i, 0]);
        file_name = url.split('/')[-1]
        file_path = os.path.join(image_folder,file_name)
        
        if os.path.isfile(file_path) and not force_download:
            continue

        response = requests.get(url)
        if response.status_code:
            with open(file_path, 'wb') as fp:
                fp.write(response.content)
        else:
            print (f"\t WARNING: File { file_name } could not be downloaded")
