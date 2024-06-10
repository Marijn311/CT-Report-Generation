import pandas as pd
import run_sarle
import os

"""
This is the main script to run SARLE-tool.
This script is an adaption of https://github.com/rachellea/sarle-labeler.
SARLE stand for Senternce Analysis for Radiology Label Extraction.
It is a tool to mine classification labels from radiology reports. 
Classification labels describe the presence or absence of abnormalities in the reports.

SARLE have two variants: SARLE-Rules and SARLE-Hybrid:
   -SARLE-Rules works on a simple principle. 
    SARLE-Rules, is a fully rule-based search algorith that is easy to understand and customise.
    There are two phases to SARLE-Rules: 1) sentence classification and 2) term search.
    In phase 1 a rule-based system is used to identify phrases that are medically “normal”.
    We define a sentence as “normal” if it describes normal findings, e.g.,
    “the lungs are clear,” or the lack of abnormal findings, e.g., “no
    masses.” We define a sentence as “abnormal” if it describes the
    presence of abnormal findings, e.g., “pneumonia in the right lung”.
    After the sentence classification step (phase 1), a rule-based "term search" (phase 2) is applied on the "abnormal" sentences.
    In the term search, a medical vocabulary and set of rules is applied to the
    “abnormal” sentences to determine exactly which abnormal findings (and locations) are present. 

   -SARLE-Hybrid differs from SARLE-Rules in that it uses a machine learning model to classify sentences as normal or abnormal in phase 1.
    This approach require labeled data for training the model.

SARLE also has the option to validate the label extraction accuracy when some manually labeled (ground truth) data is provided.

There are a few things that require user acion:
1.DATASET_PATH: the path to the dataset
2.DATASET_NAMES: the names of the hospitals in the dataset
3.SARLE_TRAIN_DATASET_FILENAME: the name of the file that contains the labeled dataset that is used to train the ML classifier for SARLE-Hybrid
4.SARLE_TEST_DATASET_FILENAME: the name of the file that contains the labeled dataset that is used to evaluate the label extraction performance
5.SARLE_PREDICT_DATASET_FILENAME: the name of the file that contains the unlabeled dataset for which we want to extract labels automatically
6.SARLE_VARIANT: the variant of SARLE that you want to use. Either: 'rules' or 'hybrid'

"""

DATASET_PATH = r"C:\Users\20192010\Downloads\sarle_Test"
DATASET_NAMES = ['AMPH', 'ISAL']
#DATASET_NAMES = ['AMPH', 'ISAL', 'LUMC', 'MAXI', 'RADB', 'UMG1', 'UMG2', 'VUMC', 'ZUYD']
SARLE_TRAIN_DATASET_FILENAME = "SARLE_train_dataset.xlsx"         # Put None if you don't want to use this
SARLE_TEST_DATASET_FILENAME = "SARLE_test_dataset.xlsx"        # Put None if you don't want to use this
SARLE_PREDICT_DATASET_FILENAME = None   # Put None if you don't want to use this
SARLE_VARIANT = 'hybrid' # Either 'rules' or 'hybrid'

if __name__=='__main__':

    # Define the datasets
    if SARLE_TRAIN_DATASET_FILENAME == None or SARLE_VARIANT == 'rules':
        train_data = pd.DataFrame()
    else:
        train_data = pd.read_excel(os.path.join(DATASET_PATH, SARLE_TRAIN_DATASET_FILENAME)) 
        
    if SARLE_TEST_DATASET_FILENAME == None:
        test_data = pd.DataFrame()
    else:
        test_data = pd.read_excel(os.path.join(DATASET_PATH, SARLE_TEST_DATASET_FILENAME))
    
    if SARLE_PREDICT_DATASET_FILENAME == None:
        predict_data = pd.DataFrame()
    else:
        predict_data = pd.read_excel(os.path.join(DATASET_PATH, SARLE_PREDICT_DATASET_FILENAME))       


    # Run the main SARLE script
    run_sarle.generate_labels(train_data, test_data, predict_data, sarle_variant=SARLE_VARIANT, use_other_abnormality=False, use_other_location=True)
    
    """ 
    The run_sarle script produces a single output file called all_labels that contains report and labels for the entire dataset.
    We would like to save the labels per hospital folder. 
    The code below splits the all_labels file into multiple dataframes and saves them in the seperate hospital folders.
    """
    
    all_labels = pd.read_excel(os.path.join(DATASET_PATH, "all_labels.xlsx"))
    for hospital in DATASET_NAMES:
        df = all_labels[all_labels['PseudoID'].str.contains(hospital)]
        df.to_excel(os.path.join(DATASET_PATH, hospital, "labels.xlsx"), index=False)