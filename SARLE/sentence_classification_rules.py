from rules import *

class ApplyRules(object):
    """apply the sentence classification rules to the sentences in the radiology reports.
    The rules are used to remove the normal phrases (sentence parts) from the sentences using the vocabulary and rules that we defined.
    It returns the sentences with the normal phrases removed to be used by the term search."""

    def __init__(self, data, setname):
        
        self.data = data
        self.setname = setname
        assert self.setname in ['train','test','predict']
                
        self.rules_order = MAIN_WORDS
        self.rules_def = MAIN_WORD_FUNCTIONS
    
    
        # Run rules to separate healthy and sick phrases.
        if not self.data.empty:
            self._apply_all_rules() 
            self._extract_predictions()
        
        self.data_processed = self.data
    
    def _apply_all_rules(self):
        self.data = self.data.rename(columns={'Sentence':'OriginalSentence'}) # The pass data with (sentence, filename, section) en we just rename the sentence to original sentence
        self.data['OriginalSentence'] = [' '+x+' ' for x in self.data['OriginalSentence'].values.tolist()] # Pad with word with whitespaces. important to ensure terms at beginning of words work
        self.data['Sentence']=self.data['OriginalSentence'].values.tolist() # Will contain padded version of sentence acceptable for term search
        self.data['PredLabelConservative']=1 # Assume sick unless marked healthy
            
        """ 
        This block below is what deletes certain words in the sentences.
        It is deleting all parts of the sentence that it thinks are describing "normal" findings.
        Sometimes it labels the sentence as sick even when it is healthy. 
        This happens because it removes sick parts of the sentence but if any part of the sentence remains it is labeled as sick, as a precaution. 
        e.g. input -> "There is no focal airspace disease." 
        The rules remove "no focal airspace disease" because it know this is an healthy statement.
        However, the words "there is" remain in the sentence. When any words in the sentence remain is is labeled sick (conservative approach).
        Luckily the term search (phase 2) extracts no abnormality labels from "there is". So there is no real problem.
            
        However, this means that looking at the PredLabelto determine classifcation quality might be misleading. 
        It is better to look at which part of the sentences are kept as abnormal findings. 
        """
        
        for idx in self.data.index.values.tolist():                             # Loop over all sentences in a dataset
            sent = self.data.at[idx,'OriginalSentence']                         # Get a original sentence
            for mainword in self.rules_order:                                   # Loop over all the "rule_order" words that are called mainwords. This is a list of words or phrases that signify that a normal finding is coming.
                func = self.rules_def[mainword]['function']                     # Pass the main word to "rules_def". Rules_def is a dictionary and or each mainword there is a simple rule based function that describes which part of the sentences should be deleted because we assume it to be healthy. In the "rules_ct.py" we can add new mainwords and corresponding functions.
                kwargs = self.rules_def[mainword]                               # Load the specific function(rule) that we need to apply.
                modified, sent = func(sentence=sent,mainword=mainword,**kwargs) # Actually apply the rule to the sentence. if any part of the sentence is deleted then modified is set to True.
                if modified == True: 
                    self.data.at[idx,'PredLabelConservative']=0 
                    self.data.at[idx,'Sentence'] = sent                         # If the sentence is modified then we update the sentence in the dataframe.
        
        
    def _extract_predictions(self):
        """Report overall performance and put binary labels, predicted
        labels, and predicted probabilities into <self.data>"""
        # Note that some outputs of some rules will produce empty sentences that
        # are not the empty string, e.g. ' ' or '   '. We need to turn these into
        # the empty string so that we can produce our labels based on assuming
        # that a healthy sentence is the empty string.
        # Note that ' '.join(' '.split()) produces the empty string.
        self.data['Sentence'] = [' '.join(x.split()) for x in self.data['Sentence'].values.tolist()]
        
        # Actual PredLabel should be healthy only if there is NOTHING left in the
        # Sentence column because then it means every component of the sentence
        # was deemed healthy. If any part is remaining, that part should be
        # treated as sick. 
        self.data['PredLabel'] = 1
        for idx in self.data.index.values.tolist():
            if self.data.at[idx,'Sentence']=='':
                self.data.at[idx,'PredLabel'] = 0
        
        # Rules are not probabilistic so the PredProb column is equal to the
        # PredLabel column. PredProb column is accessed in the eval functions.
        self.data['PredProb'] = self.data['PredLabel'].values.tolist()
        
     
                