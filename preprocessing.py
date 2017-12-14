import re
import nltk
import pickle

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()

#df_all_word_counts = pd.read_pickle('EN-DE_word_occurrence_sample_2281331.pkl')
#df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_377265.pkl')
#df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_377265_lem.pkl')
df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_2601244_lem.pkl')
#df_all_word_counts = pd.read_pickle('twitter_word_occurrence_sample_333769_lem_questions.pkl')

word_sets = [set(df_all_word_counts[df_all_word_counts['word_counts']==i]['word'].values) for i in range(1, 22, 1)]


def checkWordOccurrence(x, occurence):
    return set(x) & word_sets[occurence]

def checkWordOccurrenceLength(x, occurence):
    return len(set(x) & word_sets[occurence])

def checkChars(x):
    
    return re.sub(r'([^\s\w.?!]|_)+', r' ', x).lower() #What does this do again?

def remove_non_ascii(text):

    return re.sub(r'[^\x00-\x7F]+','', text)

def appendEOS(x):

    return x + ['<EOS>']

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
