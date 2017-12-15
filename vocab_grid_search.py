
# coding: utf-8
import pickle
import preprocessing
import pandas as pd
import nltk
import re

from joblib import Parallel, delayed

import numpy as np
import string

#df_all = pd.read_pickle('twitter_preprocess_pickle_stage_0_sample_2601244_grid_estimation_step_1.pkl')
df_all = pd.read_pickle('../../datasets/twitter_preprocess_pickle_stage_0_sample_2601244_word_occur.pkl')
#df_all = pd.read_pickle('../../datasets/twitter_preprocess_pickle_stage_0_sample_377265.pkl')
#df_all_1 = pd.read_pickle('../../datasets/twitter_preprocess_pickle_stage_0_sample_2601244_part_1.pkl')
#df_all_2 = pd.read_pickle('../../datasets/twitter_preprocess_pickle_stage_0_sample_2601244_part_2.pkl')

#df_all = pd.concat([df_all_1, df_all_2])

#df_all_1 = []
#df_all_2 = []

check_word_pair_0 = 'alpha_lem_Pair_0_tokens'
check_word_pair_1 = 'alpha_lem_Pair_1_tokens'

df_all = df_all[[check_word_pair_0, check_word_pair_1, 'n_' + check_word_pair_0, 'n_' + check_word_pair_1]]

processing_type='lem'

print ('calculate word count presence')

for i in range(0, 20, 1):
    print (i)
    func = lambda x: preprocessing.checkWordOccurrence(x, i)
    df_all['alpha_%s_Pair_0_tokens_%d_word' % (processing_type, i+1)] = df_all['alpha_%s_Pair_0_tokens' % processing_type].apply(func)
        
    df_all['alpha_%s_Pair_1_tokens_%d_word' % (processing_type, i+1)] = df_all['alpha_%s_Pair_1_tokens' % processing_type].apply(func)


df_all = df_all[(df_all['n_alpha_lem_Pair_0_tokens']>2) & (df_all['n_alpha_lem_Pair_1_tokens']>2)]

df_all[0:int(df_all.shape[0]/2)].to_pickle('../../datasets/twitter_preprocess_pickle_stage_0_sample_2601244_word_occur_part_1.pkl')
df_all[int(df_all.shape[0]/2):].to_pickle('../../datasets/twitter_preprocess_pickle_stage_0_sample_2601244_word_occur_part_2.pkl')

print ('Finished word presence counting')


def checkVocabSize(seq_max, rare_word_cut):

    df_all_seq_max = df_all[(df_all['n_alpha_lem_Pair_0_tokens'] <seq_max) & (df_all['n_alpha_lem_Pair_1_tokens'] <seq_max)]

    #print ('hello 0')
    for i in range(1, rare_word_cut + 1, 1):
        print (i, df_all_seq_max.shape[0])
        df_all_seq_max = df_all_seq_max[(df_all_seq_max['alpha_lem_Pair_0_tokens_%d_word' % i]==0) &                     
                                (df_all_seq_max['alpha_lem_Pair_1_tokens_%d_word' % i]==0)                                     
                   ] 

    print ('hello 1')
    Pair_0_words = set.union(*[set(i) for i in df_all_seq_max['alpha_lem_Pair_0_tokens'].values])

    print ('hello 2')
    Pair_1_words = set.union(*[set(i) for i in df_all_seq_max['alpha_lem_Pair_1_tokens'].values])

    print ('hello 3')
    print ('hello 4')
    vocab_size = len(list(Pair_0_words.union(Pair_1_words)))

    print ('hello 5')
    print (vocab_size)

    return vocab_size, df_all_seq_max.shape[0], seq_max, rare_word_cut

if __name__ == "__main__":

    print ('check vocab size')
 #   vocab_sizes = Parallel(n_jobs=2, verbose=8)(delayed(checkVocabSize)(max_seq, rare_word) for max_seq in range(10, 50) for rare_word in range(10, 21))

#    pickle.dump(vocab_sizes, open('../processed_data/vocab_size_grid_search_sample_%d_20_word_occurence_part_2.pkl' % df_all.shape[0], 'wb'))

