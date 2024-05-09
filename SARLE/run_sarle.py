import copy
import sentence_classification_rules, sentence_classification_ML
import term_search
from saving_utils import *
from visualizations import *

"""This script is the primary script for running SARLE. This scripts calls upon other scripts to perform the 4 main steps of SARLE:

1. Sentence classifcation: differentiating between medically normal and abnormal sentences.
2. Term search: extracting abnormality (and location) labels.
3. Saving the results to files: the classification labels that can be used later in a computer vision classification model. Also the of the sentece classification and term search results are saved such that they quality can be inspected.
4. Visualizing the results: showing the distribution of the extracted labels.

"""


def generate_labels(train_data, test_data, predict_data,
                    sarle_variant, use_other_abnormality, use_other_location):
    """Generate a matrix of abnormality x location labels for each
    narrative-style radiology report in the dataset.
    
    Variables:
    <sarle_variant> can be 'rules' or 'hybrid'.
        If 'rules' then a rule-based method will be used to filter out normal
        phrases and keep only abnormal phrases. When selecting 'hybrid' you 
        should provided labeled training data and a labeled test set. Then a 
        ML classifier will be trained on the training data and used to classify
        sentences as normal or abnormal.
    <use_other_abnormality> is either 'True' or 'False' which determines whether
        we create a label called "other abnormality" when we find a location but do not recognise a abnormality.
    <use_other_location> is either 'True' or 'False' which determines whether
        we create a label called "other location" when we find a abnormality but do not recognise a location.
   """
    
    
    #Set up results dirs
    setup = [sarle_variant, use_other_abnormality, use_other_location]
    _, sent_class_dir, term_search_dir = configure_results_dirs(*setup) #make a directory to store the results/output in
    

  
    # train_data = copy.deepcopy(train_data_raw)
    # test_data = copy.deepcopy(test_data_raw)
    # predict_data = copy.deepcopy(predict_data_raw)


    ######################################################################################################################
    #Step 1: Sentence/Phrase Classification------------------------------------------------------------------------------
    ######################################################################################################################
    print("\nStarting Phase 1: Sentence Classification (normal vs abnormal)\n")
    if sarle_variant == 'rules': 
        train_data = sentence_classification_rules.ApplyRules(train_data, 'train').data_processed
        test_data = sentence_classification_rules.ApplyRules(test_data, 'test').data_processed # This data has shape (nr_senteces, 9). The 9 columns are 'Label', 'OriginalSentence', 'Filename', 'Section', 'BinLabel', 'Sentence', 'PredLabelConservative', 'PredLabel', 'PredProb'
        predict_data = sentence_classification_rules.ApplyRules(predict_data, 'predict').data_processed
        print(f"Train dataset has {train_data.shape[0]} sentences")
        print(f"Test dataset has {test_data.shape[0]} sentences") 
        print(f"Predict dataset has {predict_data.shape[0]} sentences")
        # example = predict_data.iloc[0] #print the 9 columns of the first sentence in the predict dataset to show what the data looks like
        # print(f"Example of a the data in a predict datapoint:\n{example}")
        if train_data.shape[0] == 0:
            print("No train data was provided so no evaluation of phase 1 is possible.")
        #todo add sentence classification evaluation
        
        """
        The 9 columns of the precdit dataframes are:
        labels = for h for sick or healthy
        OriginalSentence = the original sentence (unpadded)
        Filename = the filename of the report from which the sentence came
        Section = the section of the report from which the sentence came, we can all set this to a random value for my custom dataset
        BinLabel = the binary label, 0 for healthy and 1 for sick
        Sentence = the sentence after whitespace padding is applied
        PredLabelConservative = This is always initialised as 1 because we assume that the sentence is sick unless it is marked healthy, then this is set to 0 (a conservative assumption)
        PredLabel = the predicted label, 0 for healthy and 1 for sick. This is similar to predlabelconservative in most cases but does soem final other check.
        PredProb = This is the same as predlabel. Since rules-based decisions are not probabilistic. PredProb column is accessed in the eval functions.
        """
    
    if sarle_variant == 'hybrid': #ML Sentence Classifier, using Fasttext 
        m = sentence_classification_ML.ClassifySentences(train_data, test_data, predict_data, sent_class_dir)
        m.run_all()
        train_data = m.train_data
        test_data = m.test_data
        predict_data = m.predict_data


    ######################################################################################################################
    #Step 2: Term Search ------------------------------------------------------------------------------------------------
    ######################################################################################################################
    print("\nStarting Phase 2: Term Search (extracting abnormality x location labels)")
    #In this section we take all the sentence parts that were marked as abnormal in phase 1 and pass them to the term search function.
    #In this function we loop over all the words in the abnormal sentences to see if any of the words are in the abnormality/location vocabulary what we defined.
    #We extract the found abnormalities and locations and save it in a dataframe.
    term_search.RadLabel(train_data, 'train', term_search_dir, sarle_variant, use_other_abnormality, use_other_location)
    term_search.RadLabel(test_data, 'test', term_search_dir, sarle_variant, use_other_abnormality, use_other_location)
    term_search.RadLabel(predict_data, 'predict', term_search_dir, sarle_variant, use_other_abnormality, use_other_location)


    ######################################################################################################################
    #Step 3: Save the results (mined tags) to files ---------------------------------------------------------------------
    ######################################################################################################################
    print("\nStarting Phase 3: Saving results to file")
    save_results(dataset=predict_data, sarle_variant=sarle_variant, term_search_dir=term_search_dir)
    print("Results succesfully saved to file")
    

    ######################################################################################################################
    #Step 4: Visualize the data and extracted labels ---------------------------------------------------------------------
    ######################################################################################################################
    print("\nStarting Phase 4: Showing extracted abnormalities in the predict dataset\n")
    generate_visualizations()

    

