"""This script contains the vocabulary to find abnormalities in the reports.
The "medically abnormal" sentences are search for either a word in the "Any" list or for a combination of a "Term1" and "Term2" word.

NOTE: The script searches for the strings defined below in the report sentences which are large strings. 
For example: If you search for 'long' it will also find 'longitudinal' and 'belonging'.
Adding whitespaces to the beginning and end of each word is neccesary to get the expected behaviour.
"""


ABNORMALITY_TERMS = {
    'pleural_effusion':{
        'Any':[' pleuravocht ', ' pleura-vocht ', ' pleuravloeistof ', ' pleura-vloeistof ', ' pleurale-vloeistof ', ' pleuraalvocht '],
        
        'Term1':[' pleura ', ' pleurale '], 
        
        'Term2':[' vocht ', ' effusie ', ' vloeistof ', ' vochtcollecties ']}, 

    
    'pulmonary_nodule':{
        'Any': [' longnodus ', ' longnodule ', ' longnodulen ', ' longnodulus ', ' longnodules ', 'longnoduli', ' longmetastase ', 
                ' longmetastasen ', ' longparenchymafwijkingen ', ' longmetastases ', ' solide longlaesie ', ' solide longlaesies ', 
                ' longtumor'], 
    
        'Term1': [' nodus ', ' node ', ' nodi ', ' noduli ', ' nodulus ', ' nodule ', ' nodulair ', ' nodulaire ', ' laesie ', 
                    ' laesies ', ' lesie ', ' lesies ', ' massa ', ' weke-delenmassa ', ' metastasen ', ' metastase ', ' weke-delenzwelling ', 
                    ' micronodus ', ' densiteit ', ' densiteiten ', ' solitaire afwijking ', ' solitaire afwijkingen ', ' solitaire laesie ', 
                    ' solitaire laesies ', ' gemetasteerde ziekte ', ' tumor ' ], 
                                        
        'Term2': [' longen ', ' long ', ' longweefsel ', ' longparenchym ', ' longparenchyma ', ' longsetting ', ' longlevel ',  
                    ' longwindow ', ' longvenster ', ' longvensters ', ' longvelden ', ' longveld ', ' longvolume ', ' longvoluminae ', 
                    ' kwab ', ' kwabben ', ' onderkwab ', ' middenkwab ', ' bovenkwab ', ' longkwab ', ' longkwabben ', ' onderlongkwab ', 
                    ' middenlongkwab ', ' bovenlongkwab ', ' onderkwabben ', ' middenkwabben ', ' bovenkwabben ', ' onderlongkwabben ', 
                    ' middenlongkwabben ', ' bovenlongkwabben ', ' lok ', ' lbk ', ' rok ', ' rbk ', ' lob ', ' lobben ', ' lobus ', 
                    ' rechterlong ', ' linkerlong ', ' rechterlongkwab ', ' linkerlongkwab ', ' rechterlongkwabben ', ' linkerlongkwabben ', 
                    ' rechteronderkwab ', ' linkeronderkwab ', ' rechtermiddenkwab ', ' linkermiddenkwab ', ' rechterbovenkwab ', 
                    ' linkerbovenkwab ', ' pleura ', ' pleuraal ', ' pleurale ', ' pulmonaal ', ' pulmonale ', ' pulmonalis ', 
                    ' intrapulmonaal ', ' intrapulmonale ', ' pulmonair ', ' intrapulmonair ']},
        }


ABNORMALITY_LIST = list(ABNORMALITY_TERMS.keys())

def return_abnormality_terms():
    return ABNORMALITY_TERMS
