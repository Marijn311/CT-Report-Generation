import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from main import DATASET_PATH
from abnormality_vocabulary import ABNORMALITY_LIST
matplotlib.rcParams.update({'font.size': 18})

"""
This script contains functions to generate visualizations of the dataset.
It is used to visualize the distribution of the labels in the dataset.
"""
    
def generate_visualizations():
    
    # Load the data that was saved    
    path = os.path.join(DATASET_PATH,'all_labels.xlsx')
    data = pd.read_excel(path)

    # Extract the MH_diseases column
    abnormalityes = data['MH_diseases']
    # Convert the string of arrays to lists
    abnormalityes = abnormalityes.apply(lambda x: eval(x))
    # Convert the lists to a 2D numpy array
    abnormalityes_np = np.array(abnormalityes.tolist())
    # Sum the columns of the 2d array to get the number of times a abnormalitye was found
    abnormalityes_count = abnormalityes_np.sum(axis=0)
    nr_rows=abnormalityes.shape[0]

    # Divide the number of times a abnormalitye was found by the number of reports to get the frequency
    abnormalityes_freq = abnormalityes_count/nr_rows
    abnormalityes_freq = np.round(abnormalityes_freq, decimals=3)
    
    posweight = (1-abnormalityes_freq)/abnormalityes_freq
    posweight = np.round(posweight, decimals=3)

    print(f"There are a total of {nr_rows} reports in the predict set.")
    print(f"The abnormalities that were searched for are {ABNORMALITY_LIST}")
    print(f"The abnormalities were extracted in the following frequency: {abnormalityes_count}")
    print(f"This means that x % of the reports has these labels. x = {abnormalityes_freq}")
    print(f"The positive weight for the loss function is (1-x)/x => {posweight}")


    x_labels = ABNORMALITY_LIST.copy()
    
    
    # Plot the abnormalitye count. takes the x-labels from the list
    # Make sure there is enough room at the bottom to display the full name of the labels
    plt.subplots_adjust(bottom=0.3)
    plt.bar(x_labels, abnormalityes_count)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title(f'Count of abnormality labels in the dataset. Number of samples = {nr_rows}')
    plt.show()

    # Make the same plot for the frequency
    plt.subplots_adjust(bottom=0.3)
    plt.bar(x_labels, abnormalityes_freq)
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.title(f'Frequency of abnormality labels in the dataset. Number of samples = {nr_rows}')
    plt.show()