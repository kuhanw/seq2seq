#!/usr/bin/env python
import numpy as np

PAD = 0
EOS = 1
UNK = 2

def generateRandomSeqBatchMajor(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    return [
            [random.randint(vocab_lower, vocab_upper-2) for digit in range(random.randint(length_from, length_to))] + [1]
                for batch in range(batch_size)]

def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)
    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)
    x = np.ones((batch_size, maxlen_x)).astype('int32') * PAD
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths

def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)
    
    x_lengths = np.array(lengths_x)
    y_lengths = np.array(lengths_y)

    maxlen_x = np.max(x_lengths)
    maxlen_y = np.max(y_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * PAD
    y = np.ones((batch_size, maxlen_y)).astype('int32') * PAD
    
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        y[idx, :lengths_y[idx]] = s_y
        
    return x, x_lengths, y, y_lengths
