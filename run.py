import sys
import json
from src.model.model import FactorAnalysisModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT_STATS_DIR = './output'

def record_stats(config, weights):
    with open(os.path.join(ROOT_STATS_DIR, config['experiment_name'], 'weights.txt'), "w") as outfile:
        outfile.write(np.array2string(weights))
        
def plot_stats(config, X, y):
    plt.figure()
    plt.plot(X, y, label="Negative Log Likelihood")
    plt.xlabel("Epochs")
    plt.ylabel("Negative Log Likelihood")
    plt.title(config['experiment_name']+" Stats Plot")
    plt.savefig(os.path.join(ROOT_STATS_DIR, config['experiment_name'], 'likelihood.png'))
    plt.close()
         
def get_factor_loadings(config):
    # If the model has not been run, run it
    if os.path.exists(os.path.join(ROOT_STATS_DIR, config['experiment_name'])) ==False:
        model = FactorAnalysisModel(config)
        X_train = pd.read_csv(config['training_data_path']).to_numpy()   # Read data
        factor_loadings, likelihoods = model.train(X_train)
        
        os.makedirs(os.path.join(ROOT_STATS_DIR, config['experiment_name']))
        plot_stats(config, np.arange(config['epochs']), likelihoods)
        record_stats(config, factor_loadings)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            config = json.load(open('config/testdata.json'))
            get_factor_loadings(config)
        
