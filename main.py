import pstNMF as pst
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support
import warnings

# Suppress all warnings if desired
warnings.filterwarnings("ignore")

class Config:
    FILENAME = "mnist_data.csv"
    RANK_1 = 2
    RANK_2 = 3
    NUM_REPLICATES = 10
    PARALLELIZE = True
    NUM_PROCESSES = 4  # Adjust as per your CPU cores

def process_k(K):
    nmf_instance = pst.staNMF(filepath=Config.FILENAME, K1=K, K2=K, replicates=Config.NUM_REPLICATES, parallel=Config.PARALLELIZE)
    nmf_instance.runNMF()
    nmf_instance.instability()
    print(f"Completed processing for k={K}")

def main():
    # 1. Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data

    # 2a. Pre-process the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 2b. Convert scaled data to DataFrame (for easier saving to CSV)
    df = pd.DataFrame(data_scaled)
    df.to_csv(Config.FILENAME, index=False)

    # 3. Applying staNMF with parallel processing
    # Create a pool of workers
    with Pool(processes=Config.NUM_PROCESSES) as pool:
        ks = range(Config.RANK_1, Config.RANK_2)  # Define K range
        pool.map(process_k, ks)

    # After all parallel processes are done, we can aggregate results and plot
    nmf_instance = pst.staNMF(filepath=Config.FILENAME, K1=Config.RANK_1, K2=Config.RANK_2, replicates=Config.NUM_REPLICATES, parallel=Config.PARALLELIZE)
    nmf_instance.plot(dataset_title="MNIST Instability Plot")

if __name__ == '__main__':
    freeze_support()
    main()
