
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np 

import create_model
import pickle
import random
import data_formatting

def encodeSent(sent):

    if type(sent) == str: sent = sent.split(' ')
    
    return [vocab_dict[word] if word in vocab_dict else 2 for word in sent]

def decodeSent(sent):
    return [inv_map[i] for i in sent]

def validate(op, feed_dict):

    for i, (e_in, dt_targ, dt_pred) in enumerate(zip(feed_dict['encoder_inputs:0'], 
                                                 feed_dict['decoder_targets:0'], 
                                                 session.run(op, feed_dict))):

        print('  sample {}:'.format(i + 1))
        #print('    enc input           > {}'.format(e_in))
        print('    enc input           > {}'.format(' '.join([inv_map[i] for i in e_in if i!=0])))

        #print('    dec input           > {}'.format(dt_targ))
        print('    dec input           > {}'.format(' '.join([inv_map[i] for i in dt_targ if i!=0])))

        #print('    dec train predicted > {}'.format(dt_pred))
        print('    dec train predicted > {}'.format(' '.join([inv_map[i] for i in dt_pred if i!=0])))
        if i >= 0: break


# In[ ]:


dataset = 'twitter'

df_all = pd.read_pickle('../processed_data/processed_data_v02_twitter_py35_seq_length_4_15_sample_134241_full.pkl')

df_all['alpha_Pair_1_encoding'] =  df_all['alpha_Pair_1_tokens'].apply(encodeSent)
df_all['alpha_Pair_0_encoding'] = df_all['alpha_Pair_0_tokens'].apply(encodeSent)

df_all['Index'] = df_all.index.values

df_all_train = df_all.sample(frac=0.90, random_state=0)

df_all_dev = df_all[df_all['Index'].isin(df_all_train['Index'].values) == False]

df_all_test = df_all_dev.sample(frac=0.10, random_state=0)

df_all_dev = df_all_dev[df_all_dev['Index'].isin(df_all_test['Index'].values) == False]


# In[ ]:


training_data = data_formatting.prepare_train_batch(df_all_train['alpha_Pair_0_encoding'].values, 
                                                    df_all_train['alpha_Pair_1_encoding'].values)

dev_data = data_formatting.prepare_train_batch(df_all_dev['alpha_Pair_0_encoding'].values, 
                                                    df_all_dev['alpha_Pair_1_encoding'].values)

test_data = data_formatting.prepare_train_batch(df_all_test['alpha_Pair_0_encoding'].values, 
                                                    df_all_test['alpha_Pair_1_encoding'].values)


# In[2]:


vocab_dict = pickle.load(open('../processed_data/word_dict_v02_twitter_py35_seq_length_4_15_sample_134241_full.pkl', 'rb'))
inv_map = {v: k for k, v in vocab_dict.items()}
inv_map[-1] = 'NULL'


# In[3]:


model_params = {'n_cells':128, 'num_layers':2, 'embedding_size':1024, 
          'vocab_size':len(vocab_dict) + 1, 'minibatch_size':128, 'n_threads':128,
         # 'vocab_size':20 + 1,           
          'beam_width':10, 'encoder_output_keep':0.95, 'decoder_output_keep':0.95,
         }


# In[4]:


training_params = { 'vocab_lower':3, 'vocab_upper':model_params['vocab_size']-1, 
                    'n_epochs':10000, 'batches_in_epoch':1000}
                   #'batches_in_epoch':int(df_all_train.shape[0]/model_params['minibatch_size'])}


# In[6]:


tf.reset_default_graph()

train_model = create_model.Model(model_params, 'train')

global_step = tf.Variable(0, trainable=False)

starter_learning_rate = tf.placeholder(tf.float32,shape=(),name='starter_learning_rate')

#starter_learning_rate = 0.01

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,                                    training_params['n_epochs']*training_params['batches_in_epoch'], 0.00001, staircase=False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(train_model.loss, global_step=global_step)


# In[ ]:


print_interval = 100
save_interval = 1000

lr = 0.0001

train_loss = []
dev_loss = []
learning_rate = []

#n_batch_size  = training_params['minibatch_size']

init = tf.global_variables_initializer()

with tf.Session() as session:
    
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    coord = tf.train.Coordinator()
   
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(session, 'chkpt/seq2seq_twitter_queue-21004')

    for epoch in range(training_params['n_epochs']*10):
        
        session.run(train_op, feed_dict={'starter_learning_rate:0':lr})
        
        if epoch % print_interval == 0: 
        
            print ('epoch:%d, global_step:%s, learning rate:%.3g' % 
                       (epoch, tf.train.global_step(session, global_step), 
                        session.run(optimizer._lr, feed_dict={'starter_learning_rate:0':lr})))

            train_minibatch_loss = session.run(train_model.loss, feed_dict={'starter_learning_rate:0':lr})

            train_loss.append([tf.train.global_step(session, global_step), train_minibatch_loss])  

            print ('training minibatch loss:%.6g' % (train_minibatch_loss))

        #df_sample = df_all_dev.sample(n=n_batch_size, random_state=tf.train.global_step(session, global_step))

        #input_batch_data = df_sample['alpha_Pair_0_encoding'].values
        #target_batch_data = df_sample['alpha_Pair_1_encoding'].values

        #fd_dev = data_formatting.prepare_train_batch(input_batch_data, target_batch_data)

        #feed_dict_dev = {'encoder_inputs:0': fd_dev[0],
        #                 'encoder_inputs_length:0': fd_dev[1],
        #                 'decoder_targets:0': fd_dev[2],
        #                 'decoder_targets_length:0': fd_dev[3]}

        #dev_minibatch_loss = session.run(train_model.loss, feed_dict_dev)

        #validate(train_model.decoder_pred_train, feed_dict_dev) 

        #print ('dev minibatch loss:%.6g' % (dev_minibatch_loss))

        #dev_loss.append([tf.train.global_step(session, global_step), dev_minibatch_loss])
        
        #learning_rate.append([tf.train.global_step(session, global_step), session.run(optimizer._lr)])
        
        if (epoch % save_interval == 0) & (epoch!=0): 

            saver.save(session, 'chkpt/seq2seq_twitter_queue', global_step = tf.train.global_step(session, global_step))

            print ('Session saved')
    
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)


# In[7]:


tf.reset_default_graph()


# In[8]:


inf_model = create_model.Model(model_params, 'infer')


# In[45]:


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
   
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(session, 'chkpt/seq2seq_twitter_queue-12001')
    
    #inf_out = session.run([inf_model.decoder_pred_decode, inf_model.decoder_pred_decode_prob], feed_dict_inf)
    inf_out = session.run([inf_model.encoder_inputs, inf_model.decoder_targets, inf_model.decoder_pred_decode])
    
    encoder_input = inf_out[0]
    decoder_target = inf_out[1]
    decoder_inference = inf_out[2]
    
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)
    for idx, (e_in, dt_in) in enumerate(zip(encoder_input, decoder_target)):
                
        print ('###%d###' % idx)
        
        print ('e_in', decodeSent(e_in))
        print ('d_in', decodeSent(dt_in))        
        beam_inf = list(zip(*decoder_inference[idx]))
        for idx_inf, dt_inf in enumerate(beam_inf):

            print ('dt_inf', decodeSent(dt_inf))
            if idx_inf>2: break
                

