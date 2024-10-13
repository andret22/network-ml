from models.random_forest import random_forest
import os
import joblib

# Checar para modelos salvos em cache
cached_files = [file for file in os.listdir('cache/random_forest') if file.endswith('.joblib')]

if len(cached_files) == 0:
    random_forest()

# Carregar os modelos salvos em cache
for file in cached_files:
    rf_classifier = joblib.load(f'cache/random_forest/{file}')
    

