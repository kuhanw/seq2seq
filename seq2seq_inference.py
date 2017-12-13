#!/usr/bin/env python

import tensorflow as tf
import pandas as pd
import numpy as np 

import create_model
import pickle

tf.reset_default_graph()

with tf.variable_scope('training_model'):

    inf_model = create_model.Model(dev_model_params, 'infer', test_data[:20])

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
   
    threads = tf.train.start_queue_runners(coord=coord)
    #saver.restore(session, 'd:\coding\seq2seq_2\seq2seq\chkpt\seq2seq_twitter_queue-1')
    saver.restore(session, '../chkpt/seq2seq_twitter_testing-1')
    
    #inf_out = session.run([inf_model.decoder_pred_decode, inf_model.decoder_pred_decode_prob], feed_dict_inf)
    inf_out = session.run([inf_model.encoder_inputs, inf_model.decoder_pred_decode],
            feed_dict = {'training_model/encoder_inputs:0':[[4,5,6,7]], 'training_model/encoder_inputs_length:0':[4]})
    
    encoder_input = inf_out[0]
    #decoder_target = inf_out[1]
    decoder_inference = inf_out[1]
    
    coord.request_stop()
    coord.join(threads)

    for idx, e_in in enumerate(encoder_input):
                
        print ('###%d###' % idx)
        
        print ('e_in', decodeSent(e_in))
        #print ('d_in', decodeSent(dt_in))        
        beam_inf = list(zip(*decoder_inference[idx]))
        for idx_inf, dt_inf in enumerate(beam_inf):

            print ('dt_inf', decodeSent(dt_inf))
            if idx_inf>1: break                

