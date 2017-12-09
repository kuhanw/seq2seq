import re
import nltk
import pickle

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()

#df_all_word_counts = pd.read_pickle('EN-DE_word_occurrence_sample_2281331.pkl')
df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_377265.pkl')

once_words = set(df_all_word_counts[df_all_word_counts['word_counts']==1]['word'].values)
twice_words = set(df_all_word_counts[df_all_word_counts['word_counts']==2]['word'].values)
thrice_words = set(df_all_word_counts[df_all_word_counts['word_counts']==3]['word'].values)
quad_words = set(df_all_word_counts[df_all_word_counts['word_counts']==4]['word'].values)
quint_words = set(df_all_word_counts[df_all_word_counts['word_counts']==5]['word'].values)
sext_words = set(df_all_word_counts[df_all_word_counts['word_counts']==6]['word'].values)
sept_words = set(df_all_word_counts[df_all_word_counts['word_counts']==7]['word'].values)
oct_words = set(df_all_word_counts[df_all_word_counts['word_counts']==8]['word'].values)
nine_words = set(df_all_word_counts[df_all_word_counts['word_counts']==9]['word'].values)
ten_words = set(df_all_word_counts[df_all_word_counts['word_counts']==10]['word'].values)

def checkChars(x):
    
    return re.sub(r'([^\s\w.?!]|_)+', r' ', x).lower()

def remove_non_ascii(text):

    return re.sub(r'[^\x00-\x7F]+','', text)

def appendEOS(x):

    return x + ['<EOS>']

def checkWord(x):
    
    return 'performancedriven' in x

def countTokens(x): 
    
    return len(nltk.word_tokenize(str(x)))

def nltkStem(words):
    stem_sent = [lancaster_stemmer.stem(word) for word in words] 
    return stem_sent, len(stem_sent)

def nltkLem(words):
    lem_sent = [wordnet_lemmatizer.lemmatize(word) for word in words] 
    return lem_sent, len(lem_sent)


def checkAlphaLower(words):
    
    words = nltk.word_tokenize(words)
    
    checked_text = \
        ' '.join([''.join([char for char in word if char.isalpha() or char=='?']) for word in words])
    checked_text = nltk.word_tokenize(checked_text)
    
    tokens = appendEOS([i.lower() for i in checked_text])
       
    return tokens, len(tokens)

def reverseOrdering(x):
    return [x[len(x)-i-1] for i in range(len(x))]

def createDict(word, count):
    if word!='<EOS>':
        return {word : count + 3} 
    else:
        return {word:1}
    
def makeSet(x):
    return set(x)

def checkOnceWord(x):
    return len(set(x) & once_words)

def checkTwiceWord(x):
    return len(set(x) & twice_words)

def checkThriceWord(x):
    return len(set(x) & thrice_words)

def checkQuadWord(x):
    return len(set(x) & quad_words)

def checkQuintWord(x):
    return len(set(x) & quint_words)

def checkSextWord(x):
    return len(set(x) & sext_words)

def checkSeptWord(x):
    return len(set(x) & sept_words)

def checkOctWord(x):
    return len(set(x) & oct_words)

def checkNineWord(x):
    return len(set(x) & nine_words)

def checkTenWord(x):
    return len(set(x) & ten_words)