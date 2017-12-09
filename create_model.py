#!/usr/bin/env python

import tensorflow as tf
import math
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

import data_formatting
import pandas as pd
import numpy as np


class Model:
    
    def __init__(self, params, mode, sequence_data):

        self.params = params
        self.mode = mode

        self.n_cells = params['n_cells']
        self.num_layers = params['num_layers']
        self.embedding_size = params['embedding_size']
       
        self.vocab_size = params['vocab_size']
        self.n_threads = params['n_threads']
        
        self.minibatch_size = params['minibatch_size']
        
        self.beam_width = params['beam_width']
        
        self.encoder_output_keep = params['encoder_output_keep']

        self.decoder_output_keep = params['decoder_output_keep']

        if mode == 'train':

            self.create_queue(sequence_data)
            self.initialize_placeholders()
            self.create_embeddings()
            self.create_encoder()
            self.create_training_decoder()
            self.create_training_module()            
        
        if mode == 'infer':
            
            self.create_queue(sequence_data)
            self.initialize_placeholders()
            self.create_embeddings()
            self.create_encoder()
            self.create_inference_decoder()

    def create_queue(self, training_data):
        
        encoder_data = tf.convert_to_tensor(np.asarray(training_data[0]))
        encoder_length_data = tf.convert_to_tensor(np.asarray(training_data[1]))

        decoder_data = tf.convert_to_tensor(np.asarray(training_data[2]))
        decoder_length_data = tf.convert_to_tensor(np.asarray(training_data[3]))
        
        # Note that the FIFO queue has still a capacity of 3
        #queue = tf.FIFOQueue(capacity=10, dtypes=[tf.int32, tf.int32, tf.int32, tf.int32], 
        #            shapes=[self.encoder_data.get_shape().as_list()[1:],
        #                         self.encoder_length_data.get_shape().as_list()[1:],
        #                         self.decoder_data.get_shape().as_list()[1:],
        #                         self.decoder_length_data.get_shape().as_list()[1:]
        #                        ])
        
        
        queue = tf.RandomShuffleQueue(capacity=100000, dtypes=[tf.int32, tf.int32, tf.int32, tf.int32],         
                    shapes=[encoder_data.get_shape().as_list()[1:],
                                 encoder_length_data.get_shape().as_list()[1:],
                                 decoder_data.get_shape().as_list()[1:],
                                 decoder_length_data.get_shape().as_list()[1:]
                                ], 
                                      min_after_dequeue=self.minibatch_size*100)
                                                                    
        enqueue_op = queue.enqueue_many((encoder_data, encoder_length_data, decoder_data, decoder_length_data))
        
        numberOfThreads = self.n_threads
        
        qr = tf.train.QueueRunner(queue, [enqueue_op] * numberOfThreads)
        
        tf.train.add_queue_runner(qr)
        
        self.encoder_inputs, self.encoder_inputs_length, self.decoder_targets, self.decoder_targets_length = \
                                queue.dequeue_many(n=self.minibatch_size) 
            
    def create_training_module(self):
        
        # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
        self.masks = tf.sequence_mask(lengths=self.decoder_train_length, 
                         maxlen=self.max_decoder_length, dtype=tf.float32, name='masks')

        # Computes per word average cross-entropy over a batch
        # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_outputs_train_logit, 
                                                     targets=self.decoder_train_targets,
                                                     weights=self.masks,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True)
        

    def create_inference_decoder(self):

        #if self.beam_width == 1:
            
            #encoder_last_state =self.encoder_last_state

            #encoder_outputs = self.encoder_outputs

            #encoder_inputs_length = self.encoder_inputs_length
        
        if self.beam_width != 1:
            
            self.encoder_last_state = tf.contrib.seq2seq.tile_batch(self.encoder_last_state, self.beam_width)
            
            self.encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, multiplier=self.beam_width)

            self.encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_width)
        
        self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                    num_units=self.n_cells, 
                                    memory=self.encoder_outputs,
                                    memory_sequence_length=self.encoder_inputs_length) 

        decoder_cell_list = []

        for layer in range(self.num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.n_cells, state_is_tuple=True), input_keep_prob=1, 
                                                            output_keep_prob=self.decoder_output_keep)
            decoder_cell_list.append(cell)

        #Last layer of decoders is wrapped in attention
        self.decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
                                         cell=decoder_cell_list[-1],
                                         attention_mechanism=self.attention_mechanism,
                                         attention_layer_size=self.n_cells,
                                         initial_cell_state=self.encoder_last_state[-1],                   
                                         name='Attention_Wrapper')        
       
        self.initial_state = [state for state in self.encoder_last_state]

        self.initial_state[-1] = decoder_cell_list[-1].zero_state(batch_size=tf.shape(self.encoder_outputs)[0], dtype=tf.float32)

        self.decoder_initial_state = tuple(self.initial_state)

        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
   
        start_tokens = tf.ones([self.batch_size,], tf.int32) * data_formatting.EOS

        end_token = data_formatting.EOS
    
        if self.beam_width  == 1:
            # Helper to feed inputs for greedy decoding: uses the argmax of the output
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                        end_token=end_token,
                                                        embedding=self.embedding_matrix)

            # Basic decoder performs greedy decoding at each time step
            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                 helper=decoding_helper,
                                                 initial_state=self.decoder_initial_state,
                                                 output_layer=self.output_layer)
        else:
            
            self.inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.decoder_cell,
                                                                           embedding=self.embedding_matrix,
                                                                           start_tokens=start_tokens,
                                                                           end_token=end_token,
                                                                           initial_state=self.decoder_initial_state,
                                                                           beam_width=self.beam_width,
                                                                           output_layer=self.output_layer)

        max_decode_step = tf.reduce_max(self.encoder_inputs_length) + 3

        (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (tf.contrib.seq2seq.dynamic_decode(
                    decoder=self.inference_decoder,
                    output_time_major=False,
                    maximum_iterations=max_decode_step))
           
        self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

            
    def create_training_decoder(self):

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                    num_units=self.n_cells, 
                                    memory=self.encoder_outputs, 
                                    memory_sequence_length=self.encoder_inputs_length) 

        decoder_cell_list = []

        for layer in range(self.num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.n_cells, state_is_tuple=True), input_keep_prob=1, 
                                                            output_keep_prob=self.decoder_output_keep)
            decoder_cell_list.append(cell)

        self.decoder_initial_state = self.encoder_last_state
        
        #Last layer of decoders is wrapped in attention
        decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
                                         cell=decoder_cell_list[-1],
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=self.n_cells,
                                         initial_cell_state=self.encoder_last_state[-1],                   
                                         name='Attention_Wrapper')
        
        initial_state = [state for state in self.encoder_last_state]

        initial_state[-1] = decoder_cell_list[-1].zero_state(batch_size=self.batch_size, dtype=tf.float32)

        self.decoder_initial_state = tuple(initial_state)

        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
       
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_train_inputs_embedded,
                                   sequence_length=self.decoder_train_length,
                                   time_major=False,
                                   name='training_helper')
        
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                           helper=training_helper,
                                                           initial_state=self.decoder_initial_state, 
                                                           output_layer=self.output_layer)
        
        (self.decoder_outputs_train, self.decoder_last_state_train, self.decoder_outputs_length_train) = \
              (tf.contrib.seq2seq.dynamic_decode(
                                                decoder=training_decoder,
                                                output_time_major=False,
                                                impute_finished=True,
                                                maximum_iterations=self.max_decoder_length
                                                )
              )
                    
        pad_size = tf.shape(self.decoder_train_targets)[1] - tf.shape(self.decoder_outputs_train.rnn_output)[1]
        
        self.decoder_outputs_train_logit = tf.pad(self.decoder_outputs_train.rnn_output, [[0, 0], [0,  pad_size], [0, 0]])

        self.decoder_pred_train = tf.argmax(self.decoder_outputs_train_logit, axis=-1, name='decoder_pred_train')
    
    def create_encoder(self):

        encoder_cell_list = []

        for layer in range(self.num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.n_cells, state_is_tuple=True), input_keep_prob=1, 
                                                    output_keep_prob=self.encoder_output_keep)
            encoder_cell_list.append(cell)

        encoder_cell =  tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
        
        self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                                                cell=encoder_cell, inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length, dtype=tf.float32,
                                                time_major=False)
        
        #return self.encoder_outputs, self.encoder_last_state
          
    def initialize_placeholders(self):
            
        #Create handles for encoder and decoders
       # self.encoder_inputs = tf.placeholder(shape=(None, None),
        #                dtype=tf.int32, name='encoder_inputs')

       # self.encoder_inputs_length = tf.placeholder(shape=(None,),
        #                dtype=tf.int32, name='encoder_inputs_length')

        self.output_layer = Dense(self.vocab_size, name='output_projection')

        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train':
                
                # required for training, not required for testing
            #self.decoder_targets = tf.placeholder(shape=(None, None),
            #                dtype=tf.int32, name='decoder_targets')

            #self.decoder_targets_length = tf.placeholder(shape=(None,),
             #               dtype=tf.int32, name='decoder_targets_length')

            #Make EOS and PAD matrices to concatenate with targets
                
            EOS_SLICE = tf.ones([self.batch_size, 1], dtype=tf.int32) * data_formatting.EOS
            PAD_SLICE = tf.ones([self.batch_size, 1], dtype=tf.int32) * data_formatting.PAD

            #Adding EOS to the beginning of the decoder targets
            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=1, name='decoder_train_inputs_concat')

            self.decoder_train_length = self.decoder_targets_length + 1
            
            #Don't think i really need this line here...
            self.decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=1, name='decoder_train_targets')

            #self.max_decoder_length = tf.reduce_max(self.decoder_train_length)
            
            self.max_decoder_length = tf.shape(self.decoder_train_targets)[1]
                                                                                              
    def create_embeddings(self):
        
        self.initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))

        #Randomly initialize a embedding vector for each term in the vocabulary
        self.embedding_matrix = tf.get_variable(name='embedding_matrix', shape=[self.vocab_size, self.embedding_size],
                                           initializer=self.initializer, 
                                           dtype=tf.float32)
        
        #Map each input unit to a column in the embedding matrix
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_inputs)
        
        if self.mode == 'train':
            
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_train_inputs)
