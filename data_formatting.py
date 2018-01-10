#!/usr/bin/env python
import numpy as np

PAD = 0
EOS = 1
UNK = 2

def translateDecoderOutput(output, inv_map):
    output = [decodeSent(output.T[i][0], inv_map) for i in range(len(output.T))]
    output_eos = [response.index('<eos>') if '<eos>' in response else response.index('<EOS>') for response in output]
    output = [' '.join(response[:output_eos[idx]]) for idx, response in enumerate(output)]
    output_unique = []
    for response in output:
        if response not in output_unique:
            output_unique.append(response)
    
    return output_unique

def encodeSent(sent, vocab_dict):

    if type(sent) == str: sent = sent.split(' ')
    
    return [vocab_dict[word] if word in vocab_dict else 2 for word in sent]


def generateRandomSeqBatchMajor(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    return [
            [random.randint(vocab_lower, vocab_upper-2) for digit in range(random.randint(length_from, length_to))] + [1]
                for batch in range(batch_size)]

def createInvMap(vocab_dict):
    inv_map = {v: k for k, v in vocab_dict.items()}
    inv_map[-1] = 'NULL'

    return inv_map

def decodeSent(sent, inv_map):
    return [inv_map[i] for i in sent if i not in [0, -1]]

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
