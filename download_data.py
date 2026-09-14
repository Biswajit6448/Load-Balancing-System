

import os
import json
import kaggle

# Load Kaggle API credentials from the kaggle.json file
with open('kaggle.json') as f:
    kaggle_credentials = json.load(f)

# Set environment variables for Kaggle API
os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
os.environ['KAGGLE_KEY'] = kaggle_credentials['key']

# Download the dataset
kaggle.api.dataset_download_files('shantanugarg274/load-balancing-dataset', unzip=True)



