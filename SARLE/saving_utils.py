import os
import pickle
import datetime
import pandas as pd
from main import DATASET_PATH
from abnormality_vocabulary import ABNORMALITY_LIST

def configure_results_dirs(sarle_variant):
    """Create and return the paths to "results" directories."""
    if not os.path.isdir('SARLE_results'):
        os.mkdir('SARLE_results')
    
    results_dir = os.path.join('SARLE_results',datetime.datetime.today().strftime('%Y-%m-%d')
                               +'_'+sarle_variant+'_otherabnormalitys')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if sarle_variant=='hybrid':
        sent_class_dir = os.path.join(results_dir, '0_sentences')
        if not os.path.isdir(sent_class_dir):
            os.mkdir(sent_class_dir)
    else:
        sent_class_dir = ''
    
    term_search_dir = os.path.join(results_dir, '1_term_search')
    if not os.path.isdir(term_search_dir):
        os.mkdir(term_search_dir)

    return results_dir, sent_class_dir, term_search_dir


def save_results(dataset, sarle_variant, term_search_dir):
    """The data that needs to be saved is partially in the "dataset" argument that is passed and partially in the pickle file which stores all the 
    abnormalityxlocation matrices. I will load both and then combine them to create 3 save files:
    
    -Phase1_results.xlsx: contains the pseudoID, original sentence, the predicted label (is the sentence marked as normal or abnormal), and the abnormal part of the sentence. This file allows us to check the quality of phase 1.
    -phase2_results.xlsx: contains the pseudoID, the abnormal part of the sentence, and the abnormality-location pairs that were extracted from the abnormal part of the sentence. This file allows us to check the quality of phase 2.
    -all_labels.xlsx: contains the pseudoID, the original report, and a multi hot encoding for all the abnormalitys. This file is used as the labels for CT-Net.
    """

    phase1_results = []
    phase2_results = []
    all_labels = []

    # Load the pickle file containing the abnormality_x_location matrix for all the reports in the predict dataset
    with open(os.path.join(term_search_dir,'predict_BinaryLabels.pkl'), 'rb') as file:
        loaded_data = pickle.load(file)
    
    # Create an iterator for the reports in the predict dataset
    values_iterator = iter(loaded_data.values())
    keys_iterator = iter(loaded_data.keys())

 
    for i in range(len(loaded_data)): 
        disxloc_matrix = next(values_iterator)
        pseudo_id = next(keys_iterator)
        dataset_rows = dataset[dataset['Filename'] == pseudo_id]
        
        ###########################################################################################################
        # Fill phase 1 results
        ############################################################################################################
       
        for k in range(len(dataset_rows.index)): 
            if sarle_variant == 'rules':
                og_sen = dataset_rows['OriginalSentence'].values[k]
            if sarle_variant == 'hybrid':
                og_sen = dataset_rows['Sentence'].values[k]
        
            pred_label = dataset_rows['PredLabel'].values[k]
            abnormal_part = dataset_rows['Sentence'].values[k]
            phase1_results.append([pseudo_id, og_sen, pred_label, abnormal_part])
        
        ###########################################################################################################
        # Fill phase 2 results
        ############################################################################################################
        
        abnormality_x_locations = []
        for abnormality, row in disxloc_matrix.iterrows():
            for location, value in row.items():
                if value == 1:
                    abnormality_x_locations.append((abnormality, location)) 

        abnormal_parts = []
        for k in range(len(dataset_rows.index)):
            abnormal_part = dataset_rows['Sentence'].values[k]
            abnormal_parts.append(abnormal_part)
        phase2_results.append([pseudo_id, abnormal_parts, abnormality_x_locations])
        
        ###########################################################################################################
        # Fill CT-net results
        ############################################################################################################
        
        if sarle_variant == 'rules':
            full_report = dataset_rows['OriginalSentence'].values 
        if sarle_variant == 'hybrid':
            full_report = dataset_rows['Sentence'].values
        
        # Extracting the abnormalies into an list 
        abnormalitys = [tup[0] for tup in abnormality_x_locations]
        mh_diseases = [1 if x in abnormalitys else 0 for x in ABNORMALITY_LIST] 
        all_labels.append([pseudo_id, full_report, mh_diseases]) 
        
    StopIteration
        
    # Convert the list of tuples to a DataFrame without index
    phase1_results = pd.DataFrame(phase1_results, columns=['PseudoID', 'original_sentence', 'normal_vs_abnormal', 'abnormal_part'], index=None)
    phase1_results.to_excel(os.path.join(term_search_dir,'Phase1_results.xlsx'))

    phase2_results = pd.DataFrame(phase2_results, columns=['PseudoID', 'abnormal_parts', 'abnormality_x_location'], index=None)
    phase2_results.to_excel(os.path.join(term_search_dir,'Phase2_results.xlsx'))
    
    all_labels = pd.DataFrame(all_labels, columns=['PseudoID', 'Verslag', 'MH_diseases'], index=None)
    all_labels.to_excel(os.path.join(DATASET_PATH,'all_labels.xlsx'), index=False)


