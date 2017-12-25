#!/usr/bin/env python

import tensorflow as tf
import math
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

import data_formatting
import pandas as pd
import numpy as np
import tf_helpers

class Model:
    
    def __init__(self, params, mode, sequence_data=None):

        self.params = params
        self.mode = mode

        if self.mode == 'train' and sequence_data is None:
            print ('Must provide sequence data for training!')
        
        self.n_cells = params['n_cells']
        self.num_layers = params['num_layers']
        self.embedding_size = params['embedding_size']
        self.vocab_size = params['vocab_size']
        self.beam_width = params['beam_width']
        self.limit_decode_steps = params['limit_decode_steps']


        if self.mode == 'train' or self.mode == 'debug':
            
            self.minibatch_size = params['minibatch_size']
            self.n_threads = params['n_threads']
            self.encoder_output_keep = params['encoder_output_keep']
            self.decoder_output_keep = params['decoder_output_keep']
            self.encoder_input_keep = params['encoder_input_keep']
            self.decoder_input_keep = params['decoder_input_keep']

            if self.mode != 'debug':
                print(self.mode)
                self.create_queue(sequence_data)
            
            self.initialize_placeholders()
            self.create_embeddings()
            self.create_encoder()
            if self.mode !='debug':
                self.create_training_decoder()
                self.create_training_module()            
        
            if self.mode == 'debug':
                self.create_inference_decoder()

        if self.mode == 'infer':
            
            self.encoder_output_keep = 1
            self.decoder_output_keep = 1
            self.encoder_input_keep = 1
            self.decoder_input_keep = 1
            
            #self.create_queue(sequence_data)
            self.initialize_placeholders()
            self.create_embeddings()
            self.create_encoder()
            self.create_inference_decoder()

    def create_cell(self):

        cell_unit = tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_cells, dropout_keep_prob=1.0)

        return cell_unit

    def create_queue(self, training_data):
                
        encoder_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[0]), tf.int32))
        encoder_length_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[1]), tf.int32))

        decoder_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[2]), tf.int32))
        decoder_length_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[3]), tf.int32))
        
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
                                ], seed=0,
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

        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_outputs_train_logit, 
                                                     targets=self.decoder_train_targets,
                                                     weights=self.masks,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True)
        
        self.accuracy = tf.contrib.metrics.accuracy(predictions=tf.to_int32(self.decoder_pred_train),
                                                    labels=self.decoder_train_targets, 
                                                    weights=self.masks)

    
    def create_inference_decoder(self):
       
        if self.beam_width != 1:
            
            self.encoder_last_state = tf.contrib.seq2seq.tile_batch(self.encoder_last_state, self.beam_width)
            
            self.encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, multiplier=self.beam_width)

            self.encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_width)
        
        self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                    num_units=self.n_cells, 
                                    memory=self.encoder_outputs,
                                    memory_sequence_length=self.encoder_inputs_length) 

        self.decoder_cell_list = []
        #At inference time there should be no dropout!
        for layer in range(self.num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(self.create_cell(), input_keep_prob=self.decoder_input_keep, 
                                                            output_keep_prob=self.decoder_output_keep)
            self.decoder_cell_list.append(cell)

        #Last layer of decoders is wrapped in attention
        self.decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
                                         cell=self.decoder_cell_list[-1],
                                         attention_mechanism=self.attention_mechanism,
                                         attention_layer_size=self.n_cells,
                                         initial_cell_state=self.encoder_last_state[-1],                   
                                         name='Attention_Wrapper')        
       
        self.initial_state = [state for state in self.encoder_last_state]

        self.initial_state[-1] = self.decoder_cell_list[-1].zero_state(batch_size=tf.shape(self.encoder_outputs)[0], dtype=tf.float32)

        self.decoder_initial_state = tuple(self.initial_state)

        self.decoder_cell = tf.contrib.rnn.MultiRNNCell(self.decoder_cell_list)
   
        start_tokens = tf.ones([self.batch_size,], tf.int32) * data_formatting.EOS

        end_token = data_formatting.EOS
    
        if self.beam_width  == 1:
            # Helper to feed inputs for greedy decoding: uses the argmax of the output
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                       end_token=end_token,
                                                                       embedding=self.embedding_matrix)

            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
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
            
        if self.limit_decode_steps == True:
            max_decode_step = tf.reduce_max(self.encoder_inputs_length) + 5
        else:
            max_decode_step = None
            
        if self.mode!='debug':
            
            (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (tf.contrib.seq2seq.dynamic_decode(
                    decoder=self.inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,
                    maximum_iterations=max_decode_step))
           
            if self.beam_width>1:
                self.decoder_pred_decode = tf.identity(self.decoder_outputs_decode.predicted_ids, name='decoder_pred_decode')
            else:
                self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output, axis=-1, name='decoder_pred_decode')

        elif self.mode=='debug':
            
            def _shape(batch_size, from_shape):
                if not isinstance(from_shape, tf.TensorShape):
                    return tensor_shape.TensorShape(None)
                else:
                    batch_size = tf.contrib.util.constant_value(tf.convert_to_tensor(batch_size, name="batch_size"))
                    return tf.TensorShape([batch_size]).concatenate(from_shape)

            def _create_ta(s, d):
                return tf.TensorArray(
                    dtype=d, size=0, dynamic_size=True, element_shape=_shape(self.batch_size, s))
            
            def condition(time, unused_outputs_ta, unused_state, unused_inputs, finished, unused_sequence_lengths):
                return tf.logical_not(tf.reduce_all(finished))
                #return tf.less(time, 50) 
            
            def step(time, outputs, state, inputs, finished, sequence_lengths):       
                
                prediction_lengths = state.lengths
                previously_finished = state.finished
                static_batch_size = tf.contrib.util.constant_value(self.batch_size)

                cell_state = state.cell_state      

                cell_inputs = nest.map_structure(lambda inp: self.inference_decoder._merge_batch_beams(inp, s=inp.shape[2:]), 
                                      inputs)

                cell_state = nest.map_structure(self.inference_decoder._maybe_merge_batch_beams, cell_state, 
                                    self.inference_decoder._cell.state_size)
                
                with tf.variable_scope('decoder'):

                    cell_outputs, next_cell_state = self.inference_decoder._cell(cell_inputs, cell_state)

                cell_outputs = nest.map_structure(lambda out: self.inference_decoder._split_batch_beams(out, out.shape[1:]), 
                                  cell_outputs)

                next_cell_state = nest.map_structure(self.inference_decoder._maybe_split_batch_beams, next_cell_state, 
                                     self.inference_decoder._cell.state_size)
                with tf.variable_scope('decoder'):

                    cell_outputs = self.inference_decoder._output_layer(cell_outputs)
                
              
                logits = cell_outputs
                step_log_probs = tf.nn.log_softmax(logits)
                step_log_probs = tf_helpers._mask_probs(step_log_probs, end_token, previously_finished)

                #Here is where we will add the anti-language model at a later date
                total_probs = tf.expand_dims(state.log_probs, 2) + step_log_probs

                vocab_size = logits.shape[-1].value or tf.shape(logits)[-1]

                lengths_to_add = tf.one_hot(
                  indices=tf.tile(tf.reshape(end_token, [1, 1]), [self.batch_size, self.beam_width]),
                  depth=vocab_size,
                  on_value=0,
                  off_value=1)
                
                add_mask = (1 - tf.to_int32(previously_finished))

                lengths_to_add = tf.expand_dims(add_mask, 2) * lengths_to_add

                new_prediction_lengths = (lengths_to_add + tf.expand_dims(prediction_lengths, 2))
                
                scores = total_probs

                scores_shape = tf.shape(scores)

                time = tf.convert_to_tensor(time, name="time")

                scores_flat = tf.cond(time > 0, lambda: tf.reshape(scores, [self.batch_size, -1]), lambda: scores[:, 0])

                num_available_beam = tf.cond(time > 0, lambda: tf.reduce_prod(scores_shape[1:]), lambda: tf.reduce_prod(scores_shape[2:]))

                # Pick the next beams according to the specified successors function
                #At this state we have already calculated all the possible combinations of the previous step beam + current beam
                #Thus we are evaluating the top N of sum of P(Current|Previous) + logP(Previous), thus any modification to the probs
                #must be performed prior to this step
                
                next_beam_size = tf.minimum(tf.convert_to_tensor(self.beam_width, dtype=tf.int32, name="beam_width"),
                                  num_available_beam)

                next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=next_beam_size)
                
                next_beam_scores.set_shape([static_batch_size, self.beam_width])
                
                word_indices.set_shape([static_batch_size, self.beam_width])
                
                next_beam_probs = tf_helpers._tensor_gather_helper(
                      gather_indices=word_indices,
                      gather_from=scores,
                      batch_size=self.batch_size,
                      range_size=self.beam_width * vocab_size,
                      gather_shape=[-1])

                #Here, word indices represents the positions of the flattened 
                #list of beams, which goes from 1 to beam_width*vocab_size
                #thus the word indices obtained are not the word indices of 
                #the 1 to N vocab but the flattened out list of logprobs from 1 to 
                #beam_width*vocab_size
                
                next_word_ids = tf.to_int32(word_indices % vocab_size)
                #The beam_ids represent which beams these word indices 
                #(and thereby log prob values belong to), as
                #we have word indices extracted out of 1 to beam_width*vocab_size
                next_beam_ids = tf.to_int32(word_indices / vocab_size)               
                
                previously_finished = tf_helpers._tensor_gather_helper(
                                              gather_indices=next_beam_ids,
                                              gather_from=previously_finished,
                                              batch_size=self.batch_size,
                                              range_size=self.beam_width,
                                              gather_shape=[-1])

                next_finished = tf.logical_or(previously_finished, tf.equal(next_word_ids, end_token))
                
                lengths_to_add = tf.to_int32(tf.not_equal(next_word_ids, end_token))
                lengths_to_add = (1 - tf.to_int32(next_finished)) * lengths_to_add
                
                next_prediction_len = tf_helpers._tensor_gather_helper(
                                              gather_indices=next_beam_ids,
                                              gather_from=state.lengths,
                                              batch_size=self.batch_size,
                                              range_size=self.beam_width,
                                              gather_shape=[-1])
        
                next_prediction_len += lengths_to_add
                
                next_cell_state = nest.map_structure(
                                  lambda gather_from: tf_helpers._maybe_tensor_gather_helper(
                                  gather_indices=next_beam_ids,
                                  gather_from=gather_from,
                                  batch_size=self.batch_size,
                                  range_size=self.beam_width,
                                  gather_shape=[self.batch_size * self.beam_width, -1]), next_cell_state)
               
                #We have to transform back into beamdecoder class before passing back as the input and out must have same struct
                #return beam_search_output, beam_search_state, next_inputs, finished
            
                beam_search_state = tf.contrib.seq2seq.BeamSearchDecoderState(
                             cell_state=next_cell_state,
                             log_probs=next_beam_probs,
                             lengths=next_prediction_len,
                             finished=next_finished)

                beam_search_output = tf.contrib.seq2seq.BeamSearchDecoderOutput(
                              scores=next_beam_scores,
                              predicted_ids=next_word_ids,
                              parent_ids=next_beam_ids)
                
                #Here we define the returns
                finished = beam_search_state.finished
                next_finished = finished
                sample_ids = beam_search_output.predicted_ids
                
                #This part is most important, look up the new seq terms and take their embedding
                #We pass that onto the decoder cell network in the next loop iteration

                next_inputs = tf.cond(tf.reduce_all(finished), lambda: self.inference_decoder._start_inputs,
                              lambda: self.inference_decoder._embedding_fn(sample_ids))
                #return time, outputs, state, inputs, finished, sequence_lengths
                
                outputs = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs, beam_search_output)
               
                next_sequence_lengths = tf.where(
                  tf.logical_and(tf.logical_not(finished), next_finished),
                      tf.fill(tf.shape(sequence_lengths), time + 1),
                  sequence_lengths)
                
                return time+1, outputs, beam_search_state, next_inputs, finished, next_sequence_lengths
                
            #Initialize the decoder
            self.time = 0
            
            self.finished, self.first_inputs, self.initial_state = self.inference_decoder.initialize()
            initial_time = tf.constant(self.time)
            #What is this for? initial_outputs_ta
            initial_state = self.initial_state
            initial_inputs = self.first_inputs
            initial_finished = self.finished
            initial_outputs_ta = nest.map_structure(_create_ta, self.inference_decoder.output_size,
                                                    self.inference_decoder.output_dtype)
            
            initial_sequence_lengths = tf.zeros_like(initial_finished, dtype=tf.int32)
            '''
            #cell_state = initial_state.cell_state      

            #cell_inputs = nest.map_structure(lambda inp: self.inference_decoder._merge_batch_beams(inp, s=inp.shape[2:]), 
             #                                    initial_inputs)

            #cell_state = nest.map_structure(self.inference_decoder._maybe_merge_batch_beams, cell_state, 
             #                                   self.inference_decoder._cell.state_size)

            #cell_outputs, self.next_cell_state = self.inference_decoder._cell(cell_inputs, cell_state)

            #cell_outputs = self.inference_decoder._output_layer(cell_outputs)

            #initial_logits = cell_outputs
            '''
            self.decode_loop = tf.while_loop(condition,
                                        step,
                                        loop_vars=[initial_time, initial_outputs_ta, initial_state, initial_inputs, 
                                                   initial_finished, initial_sequence_lengths],
                                        parallel_iterations=10,
                                        swap_memory=False)
            
            final_outputs_ta = self.decode_loop[1]
            self.final_state = self.decode_loop[2]
            self.final_inputs = self.decode_loop[3]
            
            self.final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
            self.final_sequence_lengths = self.decode_loop[5]
            self.predicted_ids = beam_search_ops.gather_tree(
                self.final_outputs.predicted_ids, self.final_outputs.parent_ids,
                sequence_length=self.final_sequence_lengths)
            #self.final_outputs, self.final_state = self.inference_decoder.finalize(self.final_outputs, 
            #                                                                  self.final_state, self.final_sequence_lengths)
            
    def create_training_decoder(self):

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                    num_units=self.n_cells, 
                                    memory=self.encoder_outputs, 
                                    memory_sequence_length=self.encoder_inputs_length) 

        self.decoder_cell_list = []

        for layer in range(self.num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(self.create_cell(), input_keep_prob=self.decoder_input_keep, 
                                                            output_keep_prob=self.decoder_output_keep)
            self.decoder_cell_list.append(cell)

        #self.decoder_initial_state = self.encoder_last_state
        
        #Last layer of decoders is wrapped in attention
        self.decoder_cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
                                         cell=self.decoder_cell_list[-1],
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=self.n_cells,
                                         initial_cell_state=self.encoder_last_state[-1],                   
                                         name='Attention_Wrapper')
        
        initial_state = [state for state in self.encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(batch_size=self.batch_size, dtype=tf.float32)

        self.decoder_initial_state = tuple(initial_state)

        self.decoder_cell = tf.contrib.rnn.MultiRNNCell(self.decoder_cell_list)
       
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_train_inputs_embedded,
                                   sequence_length=self.decoder_train_length,
                                   time_major=False,
                                   name='training_helper')
        
        self.training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                           helper=training_helper,
                                                           initial_state=self.decoder_initial_state, 
                                                           output_layer=self.output_layer)
        
        (self.decoder_outputs_train, self.decoder_last_state_train, self.decoder_outputs_length_train) = \
              (tf.contrib.seq2seq.dynamic_decode(
                                                decoder=self.training_decoder,
                                                output_time_major=False,
                                                impute_finished=True,
                                                maximum_iterations=self.max_decoder_length
                                                )
              )
                    
        pad_size = tf.shape(self.decoder_train_targets)[1] - tf.shape(self.decoder_outputs_train.rnn_output)[1]
        
        self.decoder_outputs_train_logit = tf.pad(self.decoder_outputs_train.rnn_output, [[0, 0], [0,  pad_size], [0, 0]])
        
        self.decoder_pred_train_prob = tf.nn.log_softmax(self.decoder_outputs_train_logit, name='decoder_pred_train_prob')

        self.decoder_pred_train = tf.argmax(self.decoder_outputs_train_logit, axis=-1, name='decoder_pred_train')
    
    def create_encoder(self):

        self.encoder_cell_list = []

        for layer in range(self.num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(self.create_cell(), input_keep_prob=self.encoder_input_keep, 
                                                    output_keep_prob=self.encoder_output_keep)
            self.encoder_cell_list.append(cell)

        self.encoder_cell =  tf.contrib.rnn.MultiRNNCell(self.encoder_cell_list)
        
        self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                                                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length, dtype=tf.float32,
                                                time_major=False)
          
    def initialize_placeholders(self):
            
        #Create handles for encoder and decoders
        if self.mode == 'infer' or self.mode=='debug':
            
            self.encoder_inputs = tf.placeholder(shape=(None, None),
                        dtype=tf.int32, name='encoder_inputs')

            self.encoder_inputs_length = tf.placeholder(shape=(None,),
                        dtype=tf.int32, name='encoder_inputs_length')

        self.output_layer = Dense(self.vocab_size, name='output_projection')

        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train' or self.mode=='debug':
            
            if self.mode == 'debug':
            # required for training, not required for testing
                self.decoder_targets = tf.placeholder(shape=(None, None),
                           dtype=tf.int32, name='decoder_targets')

                self.decoder_targets_length = tf.placeholder(shape=(None,),
                          dtype=tf.int32, name='decoder_targets_length')

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

        if self.mode == 'train' or self.mode=='debug':

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_train_inputs)
