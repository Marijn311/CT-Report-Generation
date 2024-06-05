import os
import pandas as pd
import docx 

"""
This file takes radiology reports that are stored in Word files and outputs the preprocessed radiology reports 
in an excel file that is usable by the main SARLE script.

The Excel file will have the following columns: Sentence, Filename, Section
The sentence column contains the sentences from the reports.
The Filename column contains the pseudoID of the report from which that sentence came.
The Section column is arbitrary but the SARLE tool requires this column so we just put 'Findings' for every sentence.

There are a few things that require user acion:
1. Make sure the formatting in the docx file is correct. Two enters before starting a new report title and 1 enter between the report title and the report text
2. Set the DATASET_PATH to the path where the dataset is stored
3. Set the SET_TYPE to the type of dataset you want to create. Options: 'train', 'test', 'predict'
4. Make sure the formatting in the docx file is correct. A reports starts with the pseudoID, there has to be 1 "enter" between the pseudoID and the report text. After the report text there have to be 2 "enters" before the next pseudoID.
"""

DATASET_PATH = r"C:\Users\20192010\Downloads\sarle_Test"
SET_TYPE = 'predict' # Either: 'train', 'test', 'predict'

# To store the pseudo_ids and reports 
pseudo_ids = []
reports = []
        
# Loop over all the hospital datasets 
for root, hospital_folders, _ in os.walk(DATASET_PATH):
    for hospital_folder in hospital_folders: 
        hospital_reports_doc_path = os.path.join(root, hospital_folder, 'reports.docx') 
        
        # Open the docx file and read the reports
        reports_doc = docx.Document(hospital_reports_doc_path)
        reports_text = [paragraph.text for paragraph in reports_doc.paragraphs]

        # Check if every third paragraph is an empty string (this is checking if the formating is done correctly)
        for i in range(2, len(reports_text), 3):
            assert reports_text[i] == "", f"Paragraph {i} is not empty. It contains: {reports_text[i]}"
        
        # Save the pseudo_ids and reports to the lists
        # Paragraphs 0, 3, 6, 9, 12 etc contain the pseudo_ids
        # Paragraphs 1, 4, 7, 10, 13 etc contain the report
        for i in range(0, len(reports_text), 3):
            pseudo_ids.append(reports_text[i])
            reports.append(reports_text[i+1])

        assert len(pseudo_ids) == len(reports), f"The length of the lists are not the same. The length of the pseudo_ids is {len(pseudo_ids)} and the length of the reports is {len(reports)}"
        
    break 

# Save list of all pseudo ids and reports to an excel file
df = pd.DataFrame(list(zip(pseudo_ids, reports)), columns=["PseudoID", "Report"])
all_reports_path = os.path.join(DATASET_PATH, "all_reports.xlsx")
df.to_excel(all_reports_path, index=False)

# Load the excel file with the data again
raw_df = pd.read_excel(all_reports_path, sheet_name='Sheet1')

# Format the psuedoIDs correctly
all_report_sentences = []
for index, row in raw_df.iterrows():
    psuedo_id = row['PseudoID']   
    psuedo_id = psuedo_id.strip()                               # Remove whitespace(s) at the begining and end the psuedo_id
    SplitPseudoID = psuedo_id.split('_')                        # Split the psuedo_id on the underscore
    SplitPseudoID[-1] = SplitPseudoID[-1].zfill(3)              # Make sure the id_number (last item) has 3 digits
    psuedo_id = '_'.join(SplitPseudoID)                         # Join the split psuedo_id back together
    raw_df.at[index, 'PseudoID'] = psuedo_id                    # Update the psuedo_id in the dataframe so it is the formatted version
raw_df.to_excel(all_reports_path, index=False, header=True) 


# Preprocess the reports. 
# It is very important that all the sentence are lower case, 
# that there is a trailing whitespace on each sentence, 
# and that the final period is removed. 
# Else the SARLE tool will not work propperly.
all_report_sentences = []
for index, row in raw_df.iterrows():
    report = row['Report']
    psuedo_id = row['PseudoID']

    sentences = report.split('. ')                                          # Split the report into sentences when there is a period followed by a whitespace
    sentences = [s for s in sentences if len(s)>0]                          # Remove the empty sentences that have length 0
    sentences = [s.lower() for s in sentences]                              # Make all the sentences lowercase
    sentences = [s[:-1] if s[-1] in [' ', '.'] else s for s in sentences]   # Keep removing the last character from each sentence if it is a period or a whitespace
    sentences = [s[1:] if s[0] in [' ', '.'] else s for s in sentences]     # Keep removing the first character from each sentence if it is a period or a whitespace
    sentences = [' '+s+' ' for s in sentences]                              # Add a single space to the beginning and end of each sentence. This format is required by the SARLE tool
    sentences = [s.replace(';', ',') for s in sentences]                    # Replace every ; for a ,
    sentences = [s.replace(',', ' , ') for s in sentences]                  # Make sure every comma is preceded and followed by a whitespace. This format is required by the SARLE tool
    
    
    # Put the processed sentences in a format that is compatible with the SARLE dataset: [sentence, filename, section]
    for sentence in sentences:
        all_report_sentences.append([sentence, psuedo_id, 'Findings'])


# Save the SARLE dataset to a excel file
print(f'We have {len(all_report_sentences)} sentences for in the SARLE dataset.')
processed_df = pd.DataFrame(all_report_sentences, columns = ['Sentence', 'Filename', 'Section'])
processed_df.to_excel(os.path.join(DATASET_PATH, f'SARLE_{SET_TYPE}_dataset.xlsx'), index=False, header=True)
print(f'Saved the SARLE dataset to a excel file in {DATASET_PATH}.')

print('BEWARE! if you selected a SET_TYPE of "train" or "test" you need to manually label the dataset.')
print('You can do this by opening the excel file and adding a column named "Label" which should contain a "s" or "h" for a sick or healthy sentence.')
print('You also need to add a column named "BinLabel" which should contain a 1 for a sick or 0 for a healthy sentence. This is redundant but required by the SARLE tool.')
print('For the "test" dataset you also need to add a column for every abnormality in abnormality_vocabulary.py. The column should have the exact name of the abnormality and contain a 1 if the abnormality is present in the sentence and a 0 if it is not present.')