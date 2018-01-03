import re
import nltk
import pickle

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer

wordnet_lemmatizer = WordNetLemmatizer()

lancaster_stemmer = LancasterStemmer()
porter_stemmer = PorterStemmer()

#df_all_word_counts = pd.read_pickle('EN-DE_word_occurrence_sample_2281331.pkl')
#df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_377265.pkl')
#df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_377265_lem.pkl')
df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_2601244_lem.pkl')
#df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_333769_lem_questions.pkl')

def checkWordOccurrence(x, occurence, word_set):
    return set(x) & word_set

def checkWordOccurrenceLength(x, occurence, word_set):
    return len(set(x) & word_set)

def checkChars(x):
    
    return re.sub(r'([^\s\w.?!]|_)+', r' ', x).lower() #What does this do again?

def remove_non_ascii(text):

    return re.sub(r'[^\x00-\x7F]+','', text)

def appendEOS(x):

    return x + ['<EOS>']

def countTokens(x): 
    
    return len(nltk.word_tokenize(str(x)))

def nltkStem(words):
    stem_sent = [porter_stemmer.stem(word) for word in words] 
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

def makeSet(x):
    return set(x)

def nltkNGram(text, n_gram):
    return list(nltk.ngrams(text, n_gram))
