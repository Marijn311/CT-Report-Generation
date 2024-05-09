
"""
This script contains all the functions that are used in the rule-based approach.
These functions are used to delete the normal sections from the sentences.
This code comes straight from the SARLE-Rules repository by https://github.com/rachellea/sarle-labeler
"""

def delete_mainword(sentence, mainword, **kwargs): 
    if mainword not in sentence:
        return False, sentence
    return True, sentence.replace(mainword,'')

def delete_part(sentence, delete_part, mainword, **kwargs):
    """Delete all words in the sentence coming either <delete_part>='before'
    or <delete_part>='after'"""
    if mainword not in sentence:
        return False, sentence
    senthalved = sentence.split(mainword)
    if delete_part == 'after':
        return True, senthalved[0]
    if delete_part == 'before':
        return True, senthalved[-1]

def delete_part_until(sentence, delete_part, mainword, until_hit, **kwargs): 
    """Delete all words in the sentence coming either <delete_part>='before'
    or <delete_part>='after' the <mainword> until you hit any words in the
    list <until_hit>"""
    if mainword not in sentence:
        return False, sentence
    senthalved = sentence.split(mainword)
    if delete_part == 'after':
        keep = senthalved[0] 
        dregs = senthalved[1]
        idx = len(dregs)
        for u in until_hit:
            d = dregs.find(u)
            if d < idx and d!=-1:
                idx = d
        keep2 = dregs[idx:]
        return True, keep+' '+keep2
    if delete_part == 'before':
        keep = senthalved[1]
        dregs = senthalved[0]
        idx = 0
        for u in until_hit:
            d = dregs.find(u)+len(u) 
            if d > idx and d!=-1:
                idx = d
        keep2 = dregs[0:idx]
        return True, keep2+keep 

def delete_entire_unless_immediate(sentence, mainword, position, wrange, unless_in, **kwargs):
    """Delete entire sentence if <mainword> is present, unless any of the words
    in the list <unless_in> are present within <wrange> of position=='before' or
    position=='after' the mainword in which case keep the entire sentence."""
    if mainword not in sentence:
        return False, sentence
    if position == 'after':
        if sentence.split()[-1]==mainword.strip(): 
            return True, '' 
        possible_save_words = ' '.join(sentence.split(mainword)[1].split()[0:wrange])
    elif position == 'before':
        if sentence.split()[0]==mainword.strip(): 
            return True, ''
        possible_save_words = ' '.join(sentence.split(mainword)[0].split()[-1*wrange:]) 
    saved = False
    for u in unless_in:
        if u in possible_save_words:
            saved = True    
    if saved:
        return False, sentence
    else:
        return True, ''

def delete(sentence, mainword, **kwargs): 
    """Delete entire sentence if <mainword> is present"""
    if mainword not in sentence:
        return False, sentence
    else:
        return True, ''

def delete_if_first_word(sentence, mainword, **kwargs):
    """Delete entire sentence if exactly <mainword> is the first word"""
    if mainword not in sentence:
        return False, sentence
    if mainword == sentence.split()[0]:
        return True, ''
    else:
        return False, sentence

def delete_one_before_mainword(sentence, mainword, **kwargs):
    """Delete every word starting from (and including) one word before
    <mainword>.  e.g. 'there is scarring vs
    atelectasis' -->mainword 'vs' --> 'there is' (delete both scarring and
    atelectasis)"""
    if mainword in sentence:
        s = sentence.split(mainword)[0].split()
        return True, (' ').join(s[0:-1])
    else:
        return False, sentence

def non_handling(sentence, mainword, **kwargs):
    """Delete any word that starts with 'non' or delete any word that comes
    immediately after the standalone word 'non'. Prevents the term search
    from making mistakes on words like noncalcified, nontuberculous,
    noninfectious, etc."""
    if 'non' not in sentence:
        return False, sentence
    else:
        sentlist = sentence.split()
        if ' non ' in sentence: 
            idx = sentlist.index('non')
            return True, ' '+' '.join(sentlist[0:idx]+sentlist[idx+2:])+' '
        else: 
            for word in sentlist:
                if 'non' in word:
                    sentlist.remove(word)
            return True, ' '+' '.join(sentlist)+' '

def patent_handling(sentence, mainword, **kwargs): 
    """Function for handling the word 'patent' """
    assert mainword==' patent'
    if 'patent' not in sentence:
        return False, sentence
    sentlist = sentence.split()
    if sentlist[0]=='patent':
        return delete_part_until(sentence, delete_part = 'after',mainword = 'patent', until_hit = ['status','with'])
    else: 
        return delete_part(sentence, delete_part = 'before',mainword = 'patent')

def clear_handling(sentence, mainword, **kwargs): 
    """Function for handling the word 'clear' """
    assert mainword==' clear'
    if ' clear' not in sentence:
        return False, sentence
    changed1, sentence = delete_part(sentence, delete_part='before',mainword=mainword)
    sentence = ' clear '+sentence 
    changed2, sentence = delete_part_until(sentence, delete_part='after',mainword=mainword,until_hit=['status'])
    return (changed1 or changed2), sentence

def subcentimeter_handling(sentence, mainword, **kwargs):
    """Example:
    'a few scattered subcentimeter lymph nodes are visualized not
    significantly changed from prior' --> 'a few scattered are visualized not
    significantly changed from prior'
    """
    assert mainword==' subcentimeter'
    if mainword not in sentence:
        return False, sentence
    if 'node' in ' '.join(sentence.split(mainword)[1:]):
        pre_idx = sentence.rfind(' subcentimeter')
        pre = sentence[0:pre_idx]
        post_idx = sentence.rfind('node')+len('node')
        post = sentence[post_idx:]
        sentence = pre+post
        return True, sentence
    else:
        return False, sentence