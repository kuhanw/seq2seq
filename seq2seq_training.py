#!/usr/bin/env python
import tensorflow as tf
import pandas as pd
import numpy as np 

import create_model
import pickle
import time 

import random
import data_formatting
import argparse

parser = argparse.ArgumentParser(description='Train a seq2seq model.')

parser.add_argument('-s', '--save', type = str, help = 'path to save output files.', required = True)
parser.add_argument('-r', '--restore', type = str, help = 'path to chkpt files.', required = False)
parser.add_argument('-name', '--name', type = str, help = 'Name of model, will be used as a prefix to the saved sessions.', required = True)
parser.add_argument('-cells', '--cells', type = int, help = 'Number of Hidden units.', required = True)
parser.add_argument('-n_layers', '--n_layers', type = int, help = 'Number of LSTM layers.', required = True)
parser.add_argument('-n_embedding', '--n_embedding', type = int, help = 'Dimensionality of word embedding.', required = True)
parser.add_argument('-layer_dropout', '--layer_dropout', type = float, help = 'Probability of dropout between layers, set to 1 to remove.', required = True)
parser.add_argument('-recurrent_dropout', '--recurrent_dropout', type = float, help = 'Probability of dropout between recurrent hidden states, set to 1 to remove.', required = True)
parser.add_argument('-n_epochs', '--n_epochs', type = int, help = 'Number of training epochs to run for.', required = True)
parser.add_argument('-minibatch_size', '--minibatch_size', type = int, help = 'Size of minibatch during training.', required = True)

parser.add_argument('-vocab', '--vocab', type = str, help = 'Path to vocabulary pickle file.', required = True)
parser.add_argument('-data', '--data', type = str, help = 'Path to dataset pickle file for training/testing.', required = True)

args = parser.parse_args()

print ('Arguments:', args)

chkpt_path = args.restore
save_path = args.save
vocab_path = args.vocab
data_path = args.data

def validate(train):
    
    if train == True:
        
        model = train_model
        mode = 'TRAIN'
        
    else:
        mode = 'DEV'
        model = dev_model

    encoder, decoder, predicted = session.run([model.encoder_inputs, model.decoder_targets, model.decoder_pred_train])            

    print ('Current mode:%s' % mode)
    for i, (e_in, dt_targ, dt_pred) in enumerate(zip( encoder, decoder, predicted)):

        print('  sample {}:'.format(i + 1))

        print('    enc input           > {}'.format(data_formatting.decodeSent(e_in, inv_map)))

        print('    dec input           > {}'.format(data_formatting.decodeSent(dt_targ, inv_map)))

        print('    dec train predicted > {}'.format(data_formatting.decodeSent(dt_pred, inv_map)))

        if i >= 0: break


dataset = args.name

vocab_dict = pickle.load(open(vocab_path, 'rb'))
df_all = pd.read_pickle(data_path)

df_all_train = df_all.sample(frac=0.96, random_state=1)

df_all_dev = df_all[df_all['Index'].isin(df_all_train['Index'].values) == False]

df_all_test = df_all_dev.sample(frac=0.10, random_state=1)

df_all_dev = df_all_dev[df_all_dev['Index'].isin(df_all_test['Index'].values) == False]

print ('Total Rows of Data:%d, training data:%d, dev data:%d, test_data:%d, vocab_size:%d' % (df_all.shape[0], df_all_train.shape[0],  df_all_dev.shape[0], df_all_test.shape[0], len(vocab_dict)))


train_data = data_formatting.prepare_train_batch(df_all_train['alpha_Pair_0_encoding'].values, 
                                                    df_all_train['alpha_Pair_1_encoding'].values)

dev_data = data_formatting.prepare_train_batch(df_all_dev['alpha_Pair_0_encoding'].values, 
                                                    df_all_dev['alpha_Pair_1_encoding'].values)

test_data = data_formatting.prepare_train_batch(df_all_test['alpha_Pair_0_encoding'].values, 
                                                    df_all_test['alpha_Pair_1_encoding'].values)
inv_map = data_formatting.createInvMap(vocab_dict)

train_model_params = {'n_cells':args.cells, 'num_layers':args.n_layers, 'embedding_size':args.n_embedding, 
          'vocab_size':len(vocab_dict) + 1, 'minibatch_size':args.minibatch_size, 'n_threads':128,
          'encoder_input_keep':args.layer_dropout, 'decoder_input_keep':args.layer_dropout,
          'encoder_output_keep':args.layer_dropout, 'decoder_output_keep':args.layer_dropout,
          'recurrent_dropout':args.recurrent_dropout
         }
dev_model_params = train_model_params
dev_model_params['encoder_input_keep'] = 1
dev_model_params['encoder_output_keep'] = 1
dev_model_params['decoder_input_keep'] = 1
dev_model_params['decoder_output_keep'] = 1
dev_model_params['recurrent_dropout'] = 1

training_params = { 'vocab_lower':3, 'vocab_upper':train_model_params['vocab_size']-1, 
                    'n_epochs':args.n_epochs}

tf.reset_default_graph()


with tf.variable_scope('training_model'):
    
    train_model = create_model.Model(train_model_params, 'train', train_data)

with tf.variable_scope('training_model', reuse=True):

    dev_model = create_model.Model(dev_model_params, 'train', dev_data)    

global_step = tf.Variable(0, trainable=False)

starter_learning_rate = tf.placeholder(tf.float32, shape=(), name='starter_learning_rate')

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(train_model.loss, global_step=global_step)

print_interval = 100
save_interval = 2000

lr = 0.001

metrics = []
total_time = time.time()
with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord)

    if chkpt_path is not None:
        print ('restoring from chkpt %s' % chkpt_path)
        print ('latest checkpoint %s' %  tf.train.latest_checkpoint(chkpt_path)) 
        saver.restore(session, tf.train.latest_checkpoint(chkpt_path))

    for epoch in range(training_params['n_epochs']):

        start_time = time.time()

        session.run(train_op, feed_dict={'starter_learning_rate:0':lr})
        #session.run(train_op)

        if epoch % print_interval == 0: 

            print ('epoch:%d, global_step:%s, learning rate:%.3g' % 
                       (epoch, tf.train.global_step(session, global_step), 
                        session.run(optimizer._lr, feed_dict={'starter_learning_rate:0':lr})))

            train_minibatch_loss,  train_minibatch_accuracy = session.run([train_model.loss, train_model.accuracy], 
                                                                          feed_dict={'starter_learning_rate:0':lr})

            print ('training minibatch loss:%.6g' % (train_minibatch_loss))
            print ('training minibatch accuracy:%.6g' % (train_minibatch_accuracy))

            validate(train=True)

            dev_model_loss, dev_model_accuracy = session.run([dev_model.loss, dev_model.accuracy])            

            print ('dev minibatch loss:%.6g' %  (dev_model_loss))
            print ('dev minibatch accuracy:%.6g' %  (dev_model_accuracy))

            validate(train=False)
            
            metrics.append([tf.train.global_step(session, global_step), train_minibatch_loss, train_minibatch_accuracy,
						dev_model_loss, dev_model_accuracy, session.run(optimizer._lr, feed_dict={'starter_learning_rate:0':lr})])  

            print ('Epoch:%d finished, time:%.4g' % (epoch, time.time() - start_time))

        if (epoch % save_interval == 0):# & (epoch!=0): 

            df_metrics = pd.DataFrame(metrics, columns=['Global Step','Train Loss', 'Train Accuracy', 'Dev Loss', 'Dev Accuracy', 
										'Learning Rate'])
            df_metrics.to_csv(save_path + '/training_metrics_%s.csv' % tf.train.global_step(session, global_step))
            
            saver.save(session, save_path + '/seq2seq_%s' % dataset, global_step = tf.train.global_step(session, global_step))

            print ('Session saved')

    coord.request_stop()
    coord.join(threads)

session.close()

print ('All Done!')
print ('Total time:%.4g' % (time.time() -total_time))
