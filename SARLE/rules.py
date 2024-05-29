import rule_utils as fxn

"""
This script contains the 'main word' and the rules for the rule-based approach.
The main words are words that signify that a normal finding is present in the sentence.
The rules describe how/which part of the sentence should be deleted based on that main word to remove the normal finding.
The main word dictionary contains a rule for every main word.
"""

MAIN_WORDS = ['geen', 'status na', 'status bij', 'normaal', 'normale', 'niet suspect', 'niet pathologisch', 'niet afwijkend',
              'binnen de norm', 'fysiologisch', 'fysiologische', 'postoperatief', 'postoperatieve', 'niet verdacht',
              'niet verdachte', 'niet vergroot', 'niet abnormaal', 'zonder pathologie', 'zonder pathologische', 'zonder aanwijzingen'
                ]

MAIN_WORD_FUNCTIONS = {
        'geen':{'function':fxn.delete},
        'status na':{'function':fxn.delete_part,'delete_part':'after'},
        'status bij':{'function':fxn.delete_part,'delete_part':'after'},
        'normaal':{'function':fxn.delete},
        'normale':{'function':fxn.delete},
        'niet suspect':{'function':fxn.delete},
        'niet pathologisch':{'function':fxn.delete},
        'niet afwijkend':{'function':fxn.delete},
        'binnen de norm':{'function':fxn.delete},
        'fysiologisch':{'function':fxn.delete},
        'fysiologische':{'function':fxn.delete},
        'postoperatief':{'function':fxn.delete},
        'postoperatieve':{'function':fxn.delete},
        'niet verdacht':{'function':fxn.delete},
        'niet verdachte':{'function':fxn.delete},
        'niet vergroot':{'function':fxn.delete},
        'niet abnormaal':{'function':fxn.delete},
        'zonder pathologie':{'function':fxn.delete},
        'zonder pathologische':{'function':fxn.delete},
        'zonder aanwijzingen':{'function':fxn.delete},
        }