import os
import copy
import pickle
import pandas as pd
import numpy as np
from evaluation import term_search_performance
from abnormality_vocabulary import return_abnormality_terms

"""Thisscript performs the term search on the sentences that were labeled as abnormal in phase 1.
We save the extracted abnormalities and locations in a matrix per radio report."""

###############################################################################################################
# Extract the abnormalities and locations on the full report level (not sentence level)------------------------
###############################################################################################################
class RadLabel(object):
    def __init__(self, data, setname, results_dir, 
                sarle_variant, use_other_abnormality, use_other_location, save_output_files=True):
        """Output:
        performance metrics files reporting label frequency, and performance of the different methods.
        
        Output:
        pandas dataframe with report-level labels on the test set.
        The filename is the index in this outputted dataframe
        and the columns are different classification labels.
        e.g. out.loc['12345.txt','atelectasis'] = 1 if report 12345.txt is
        positive for atelectasis (0 if negative for atelectasis.)
        Saved as files with prefix "Test_Set_Labels"
        
        Variables:
        <data> is a pandas dataframe produced by phase 1 which contains the following columns:
            ['Count','Sentence','Filename','Section','PredLabel','PredProb']
            For the test and train sets it also contains these columns:
            ['Label', 'BinLabel'] which are the ground truth sentence labels
        <setname> is train' or 'test' or 'predict', a string that will be 
            prepended to any saved files,
        <results_dir>: path to directory in which results will be saved
        <save_output_files> is True by default to ensure all output files
            are saved. It is only set to False within unit tests to avoid
            saving output files during unit testing."""
        
        self.sarle_variant = sarle_variant
        self.use_other_abnormality = use_other_abnormality
        self.use_other_location = use_other_location

        assert setname in ['train','test','predict']
        self.setname = setname
        self.results_dir = results_dir
        self.data = data
        
        self.save_output_files = save_output_files

        #Run
        if not self.data.empty:
            self.run_all()
    
    def run_all(self):
        print(f'\nWorking on term search for {self.setname} dataset.')

        #pad sentences with spaces to facilitate term search of words at 
        #the beginning and end of sentences
        self.data['Sentence'] = [' '+x+' ' for x in self.data['Sentence'].values.tolist()] 
        print(f'{self.setname} dataset has {self.data.shape[0]} sentences')
              
        #Get unique list of filenames (used in multiple methods)
        self.uniq_set_files = [x for x in set(self.data['Filename'].values.tolist())]
        self.uniq_set_files.sort()

        #Make a seperate sick and healthy sentence dataframe
        self.sickdf, self.healthydf = self.pick_out_sick_sentences()

        #Initialize dictionaries from vocab files. These dictionaries contain all the search terms used to label the abnormal sentences.
        self.initialize_vocabulary_dicts() 
        
        #Run SARLE term search step
        #This creates an outputfile which is a dictionary where the keys are filenames (report IDs), 
        # and the values are binary pandas dataframes. A single pandas dataframe is
        # organized with abnormalisities along the rows and locations alomng the columns.
        self.obtain_sarle_complex_labels()     

        #Obtain disea-only labels and binarise the outputs
        self.abnormality_out = self.binarize_complex_labels(chosen_labels=list(self.mega_abnormality_dict.keys()), label_type='abnormality')

        #Save output
        if self.save_output_files: 
            self.basic_save()


        ###########################################################################################
        # Evaluation of phase 2 performance #------------------------------------------------------
        ###########################################################################################
        #Evaluate performance of phase 2, abnormality and location detection
        if self.setname == 'test' and self.data.shape[0] != 0:
            term_search_performance(self)
        if self.setname == 'test' and self.data.shape[0] == 0:
            print('Comparison of extracted abnormalities to ground truth is NOT done because there is no labeled test set provided.')
        

    #########################################################################################################
    # Helper Functions for Extracting the Report-Level Labels -----------------------------------------------
    #########################################################################################################

    def binarize_complex_labels(self, chosen_labels, label_type):
        """Return a dataframe with index of filenames (from self.uniq_set_files)
        and columns of <chosen_labels>. This is the old-fashioned output format.
        <label_type> is the type of the chosen_labels. It is either 'abnormality'
        (to access rows) or 'location' (to access columns)"""
        assert label_type in ['abnormality','location']
        binarized = pd.DataFrame(np.zeros((len(self.uniq_set_files),len(chosen_labels))),
                        index = self.uniq_set_files, columns = chosen_labels)
        for filename in self.uniq_set_files:
            for label in chosen_labels:
                if label_type == 'location':
                    value = np.sum(self.out_bin[filename].loc[:,label])
                elif label_type == 'abnormality':
                    value = np.sum(self.out_bin[filename].loc[label,:])
                if value > 0:
                    value = 1
                binarized.at[filename, label] = value
        return binarized
    
    def obtain_sarle_complex_labels(self): 
        """Generate location and abnormality labels
        Produces self.out_bin which is a dictionary where the keys are filenames
        (report IDs), and the values are binary pandas dataframes (since I don't
        think counts make sense in this context.) A single pandas dataframe is
        organized with abnormality as rows and locations as columns."""
        self.out_bin = {}
        if self.setname == 'train':            
            other_abnormality = open(os.path.join(self.results_dir,'train_other_abnormality_sentences.txt'),'a')
            other_location = open(os.path.join(self.results_dir,'train_other_location_sentences.txt'),'a')

        #Fill self.out with dataframes of predicted labels:
        for filename in self.uniq_set_files:
            #selected_out is for this filename only:
            selected_out = pd.DataFrame(np.zeros((len(list(self.mega_abnormality_dict.keys()))+1,
                                              len(list(self.mega_loc_dict.keys()))+1)),
                           columns = list(self.mega_loc_dict.keys())+['other_location'],
                           index = list(self.mega_abnormality_dict.keys())+['other_abnormality'])     
            selected_sickdf = self.sickdf[self.sickdf['Filename']==filename]
            for sentence in selected_sickdf['Sentence'].values.tolist():
                #the temp dfs, index is keyterms and column is 'SentenceValue'
                temp_location = self.return_temp_for_location_search(sentence)
                temp_abnormality = self.return_temp_for_abnormality_search(sentence)
                
                #iterate through locations first
                for location in temp_location.index.values.tolist():
                    if temp_location.at[location,'SentenceValue'] > 0:
                        #once you know the location, figure out the abnormality
                        location_recorded = False
                        for abnormality in temp_abnormality.index.values.tolist():
                            if temp_abnormality.at[abnormality,'SentenceValue'] > 0 :
                                selected_out.at[abnormality, location] = 1
                                location_recorded = True
                        if not location_recorded:
                            #makes sure every location gets recorded
                            if self.use_other_abnormality == True:
                                selected_out.at['other_abnormality', location] = 1
                            else:    
                                selected_out.at['other_abnormality', location] = 0
                            if self.setname == 'train':
                                other_abnormality.write(location+'\t'+sentence+'\n')
                
                #iterate through abnormality second and make sure none were missed
                for abnormality in temp_abnormality.index.values.tolist():
                    if temp_abnormality.at[abnormality,'SentenceValue'] > 0:
                        if np.sum(selected_out.loc[abnormality,:].values) == 0:
                            #i.e. if we haven't recorded that abnormality yet,
                            if self.use_other_location == True:
                                selected_out.at[abnormality,'other_location'] = 1
                            else:
                                selected_out.at[abnormality,'other_location'] = 0
                            if self.setname == 'train':
                                other_location.write(abnormality+'\t'+sentence+'\n')

            self.out_bin[filename] = selected_out
        #now self.outbin should contain a dataframe for every report. Each dataframe has the abnormality as rows and the location as columns.
    
    def initialize_vocabulary_dicts(self):
        #Load dictionaries 
        self.mega_loc_dict = dict()
        self.mega_abnormality_dict = return_abnormality_terms()


    def return_temp_for_location_search(self, sentence):
        """Return a dataframe called <temp> which reports the results of
        the location term search using rules defined by <loc_dict> and
        <abnormality_dict>, for the string <sentence>
        <body_region> is 'lung' or 'heart' or 'other.' Determines how/whether
            the abnormality_dict will be used here"""
        #temp is a dataframe for this particular sentence ONLY
        temp = pd.DataFrame(np.zeros((len(list(self.mega_loc_dict.keys())),1)),
                                        index = list(self.mega_loc_dict.keys()),
                                        columns = ['SentenceValue'])
        
        #look for location phrases
        for locterm in self.mega_loc_dict.keys():
            locterm_present = RadLabel.label_for_keyterm_and_sentence(locterm, sentence, self.mega_loc_dict)
            if locterm_present: #update out dataframe for specific lung location
                temp.at[locterm,'SentenceValue'] = 1
        
        return temp
    
    
    def return_temp_for_abnormality_search(self, sentence):
        """Return a dataframe called <temp> which reports the results of
        the abnormality term search defined by <abnormality_dict> for the string <sentence>"""
        #temp is a dataframe for this particular sentence ONLY
        temp = pd.DataFrame(np.zeros((len(list(self.mega_abnormality_dict.keys())),1)),
                                        index = list(self.mega_abnormality_dict.keys()),
                                        columns = ['SentenceValue'])
        #Look for abnormality phrases
        for abnormalityterm in self.mega_abnormality_dict.keys():
            abnormalityterm_present = RadLabel.label_for_keyterm_and_sentence(abnormalityterm, sentence, self.mega_abnormality_dict)
            if abnormalityterm_present: #update out dataframe for specific lung location
                temp.at[abnormalityterm,'SentenceValue'] = 1

        return temp
        
      
    def pick_out_sick_sentences(self):
        """Separate sick sentences and healthy sentences and return
        as separate dataframes"""
        sickdf = copy.deepcopy(self.data)
        healthydf = copy.deepcopy(self.data)

        #for if you have a custom dataset that you labeled (and want to use hybrid)
        if self.sarle_variant == 'hybrid':
            sets_use_sentence_level_grtruth = ['train']
            sets_use_pred_sentence_labels = ['test', 'predict'] #we have a GT for test we should use it. only during the evaluation
        
        #for if you have a custom dataset without labels (you need to use rules)
        elif self.sarle_variant == 'rules':
            sets_use_sentence_level_grtruth = [] #no ground truth available 
            sets_use_pred_sentence_labels = ['train','test','predict']

        if self.setname in sets_use_sentence_level_grtruth:
            #BinLabel is 1 or 0 based off of Label which is 's' or 'h'
            sickdf = sickdf[sickdf['BinLabel'] == 1]
            healthydf = healthydf[healthydf['BinLabel'] == 0]
        elif self.setname in sets_use_pred_sentence_labels:
            sickdf = sickdf[sickdf['PredLabel'] == 1]
            healthydf = healthydf[healthydf['PredLabel'] == 0]
        sickdf = sickdf.reset_index(drop=True)
        healthydf = healthydf.reset_index(drop=True)
        assert (sickdf.shape[0]+healthydf.shape[0])==self.data.shape[0]
        print(f'There are {sickdf.shape[0]} abnormal sentences in {self.setname}')
        print(f'There are {healthydf.shape[0]} normal sentences in {self.setname}')
        return sickdf, healthydf

        
    def basic_save(self):
        pickle.dump(self.out_bin, open(os.path.join(self.results_dir, self.setname+'_BinaryLabels.pkl'), 'wb'))
        self.abnormality_out.to_csv(os.path.join(self.results_dir, self.setname+'_abnormalityBinaryLabels.csv'))
        self.data.to_csv(os.path.join(self.results_dir, self.setname+'_Merged.csv'))
        
    
    ###########################################################################
    # Static Methods #---------------------------------------------------------
    ###########################################################################
    @staticmethod
    def label_for_keyterm_and_sentence(keyterm, sentence, termdict):
        """Return label = 1 if <keyterm> in <sentence> else return label = 0"""
        sentence = ' ' + sentence + ' '
        label = 0
        for any_term in termdict[keyterm]['Any']:
            if any_term in sentence:
                label = 1
        if label == 0: #if label is still 0 check for secondary equivalent terms
            if 'Term1' in termdict[keyterm].keys():
                for term1 in termdict[keyterm]['Term1']:
                    for term2 in termdict[keyterm]['Term2']:
                        if (term1 in sentence) and (term2 in sentence):
                            label = 1
                            break
        #Dealing with 'Exclude'
        if 'Exclude' in termdict[keyterm].keys():
            for banned_term in termdict[keyterm]['Exclude']:
                if banned_term in sentence:
                    label = 0 # Cannot write return 0 because that means the function doesn't return any value
        return label
    
