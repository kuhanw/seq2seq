#!/usr/bin/env python

import math
import pickle
import data_formatting
import tf_helpers

from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

import numpy as np
import tensorflow as tf

class Model:
    
    def __init__(self, params, mode, sequence_data=None, ngram_model=None):

        self.params = params
        self.mode = mode

        if self.mode == 'train' and sequence_data is None:
            print ('Must provide sequence data for training!')
        
        self.n_cells = params['n_cells']
        self.num_layers = params['num_layers']
        self.embedding_size = params['embedding_size']
        self.vocab_size = params['vocab_size']

        if self.mode == 'train':
            
            self.minibatch_size = params['minibatch_size']
            self.n_threads = params['n_threads']
            self.recurrent_dropout = params['recurrent_dropout']
            self.encoder_output_keep = params['encoder_output_keep']
            self.decoder_output_keep = params['decoder_output_keep']
            self.encoder_input_keep = params['encoder_input_keep']
            self.decoder_input_keep = params['decoder_input_keep']
            
            self.create_queue(sequence_data)            
            self.initialize_placeholders()
            self.create_embeddings()
            self.create_encoder()
            
            self.create_training_decoder()
            self.create_training_module()            

        if self.mode == 'infer':
                            
            if self.anti_lm_weight!=-1 and ngram_model == None:
                print ('Must specify language model for weighting!')
                
            self.n_grams = ngram_model

            self.encoder_output_keep = 1
            self.decoder_output_keep = 1
            self.encoder_input_keep = 1
            self.decoder_input_keep = 1
            self.recurrent_dropout = 1
            
            self.beam_width = params['beam_width']        
            self.limit_decode_steps = params['limit_decode_steps']
            self.anti_lm_weight = params['anti_lm_weight']
            self.anti_lm_max_step = params['anti_lm_max_step']
            self.length_penalty = params['length_penalty']
            
            self.initialize_placeholders()
            self.create_embeddings()
            self.create_encoder()
            self.create_inference_decoder()

    def create_cell(self):
        #I believe this is recurrent dropout as opposed to dropout across layers
        cell_unit = tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_cells, dropout_keep_prob=self.recurrent_dropout)

        return cell_unit

    def create_queue(self, training_data):
                
        encoder_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[0]), tf.int32))
        encoder_length_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[1]), tf.int32))

        decoder_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[2]), tf.int32))
        decoder_length_data = tf.convert_to_tensor(tf.cast(np.asarray(training_data[3]), tf.int32))
        
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
            
            self.decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                       end_token=end_token,
                                                                       embedding=self.embedding_matrix)

            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                     helper=self.decoding_helper,
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
            
        if self.limit_decode_steps:
            self.max_decode_step = self.limit_decode_steps # + tf.reduce_max(self.encoder_inputs_length)
        else:
            self.max_decode_step = None
            
        if self.anti_lm_weight == -1:
            
            (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (tf.contrib.seq2seq.dynamic_decode(
                    decoder=self.inference_decoder,
                    output_time_major=False,
                    impute_finished=False,
                    maximum_iterations=self.max_decode_step,
                   parallel_iterations=1))
           
            if self.beam_width>1:
                self.decoder_pred_decode = tf.identity(self.decoder_outputs_decode.predicted_ids, name='decoder_pred_decode')
            else:
                self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output, axis=-1, name='decoder_pred_decode')

        else:
            
            def _shape(batch_size, from_shape):
                if not isinstance(from_shape, tf.TensorShape):
                    return tensor_shape.TensorShape(None)
                else:
                    batch_size = tf.contrib.util.constant_value(tf.convert_to_tensor(batch_size, name="batch_size"))
                    return tf.TensorShape([batch_size]).concatenate(from_shape)

            def _create_ta(s, d):
                return tf.TensorArray(
                    dtype=d, size=0, dynamic_size=True, element_shape=_shape(self.batch_size, s), clear_after_read=False)
            
            def decoder_condition(time, unused_outputs_ta, unused_state, unused_inputs, finished, unused_sequence_lengths):
                return tf.logical_not(tf.reduce_all(finished))
            
            def anti_lm_condition(test, n_grams_tf, beam_pad, current_beam, n_beams, current_beam_step):
                return current_beam < n_beams #number of beams, i.e. beam width, will specify at run time    
            
            def anti_lm_body(test, n_grams_tf, beam_pad, current_beam, n_beams, current_beam_step):
                #This loops on each beam individually
      
                def grab_probs(n_grams_tf, y_equal): 

                    y_diff_gather = tf.gather(n_grams_tf, tf.where(y_equal))

                    last_token_ids = y_diff_gather[:,0][:,-2]

                    indices = tf.reshape(last_token_ids, [tf.shape(last_token_ids)[0], 1])

                    last_token_counts = y_diff_gather[:,0][:,-1]

                    total_count = tf.reduce_sum(last_token_counts)

                    test_add_result = tf.scatter_nd(indices=indices, 
                                updates=last_token_counts, shape=scatter_base)/total_count

                    test_add_result = tf.log(test_add_result + 10e-10)

                    return tf.reshape(test_add_result, [1, self.vocab_size])

                def dump_zeros(n_grams_tf, y_equal): 

                    return tf.constant([[0. for i in range(self.vocab_size)]], dtype=tf.float64)

                scatter_base = tf.constant([self.vocab_size]) #Size of dict for scatter base

                #Find where current beam matches n_gram sequence up to current seq pos, cast as int
                matched_seqs = tf.to_int32(tf.equal(n_grams_tf, beam_pad[current_beam]))

                #Find args where the beam is matched to the n_gram combinations
                y_equal  = tf.equal(current_beam_step, tf.reduce_sum(matched_seqs[:,:current_beam_step], axis=1))

                #Get probabilities. Special case for sequences that do not match anything in the ngram model
                test_add = tf.cond(tf.equal(0, tf.reduce_sum(tf.to_int32(y_equal))),
                    true_fn=lambda: dump_zeros(n_grams_tf, y_equal),
                    false_fn=lambda : grab_probs(n_grams_tf, y_equal))

                test = tf.concat([test, test_add], axis=0)

                return test, n_grams_tf, beam_pad, current_beam+1, n_beams, current_beam_step
            
            def get_scores_lm(log_probs, sequence_lengths, length_penalty_weight, time):

                def fn1(): return tf.constant(self.n_grams[1], dtype=tf.int32)
                def fn2(): return tf.constant(self.n_grams[2], dtype=tf.int32)
                def fn3(): return tf.constant(self.n_grams[3], dtype=tf.int32)

                def fn_default(): return tf.constant(-1, dtype=tf.int32)

                def time_zero_anti_lm(score):
                    
                    n_grams_tf = tf.constant(self.n_grams[0], dtype=tf.int32)
                    
                    scatter_base = tf.constant([self.vocab_size]) #Size of dict for scatter base will specify at run time

                    token_counts = n_grams_tf[:,1]

                    total_count = tf.reduce_sum(token_counts)
                    
                    token_ids = n_grams_tf[:,0]

                    indices = tf.reshape(token_ids, [tf.shape(token_ids)[0], 1])

                    test_add = tf.scatter_nd(indices=indices, 
                                updates=token_counts, shape=scatter_base)/total_count

                    #Add small value for numerically stability
                    test_add = tf.log(test_add + 10e-10)    

                    test_add_tile = tf.reshape(tf.tile(test_add, multiples=[self.beam_width]), shape=[1, self.beam_width, self.vocab_size])

                    score = score - self.anti_lm_weight*tf.to_float(test_add_tile)

                    return score

                def time_not_zero_anti_lm(score):
                    #anti-lm correction up to the 4th seq position at most
                    n_grams_tf = tf.case(
                                {tf.equal(time,1): fn1, 
                                 tf.equal(time,2): fn2,
                                 tf.equal(time,3): fn3}, default=fn_default, exclusive=True)        

                    concat_base = tf.constant([[1.0 for i in range(self.vocab_size)]], dtype=tf.float64) 

                    beam = tf.transpose(initial_outputs_ta.predicted_ids.concat())

                    beam_pad = tf.pad(beam, [[0, 0], [0, 2]], mode='CONSTANT', constant_values=0)
                    #Start the anti-lm loop from from the first beam
                    initial_beam_step = tf.constant(0)  

                    n_beams = tf.constant(self.beam_width) 

                    anti_lm_outputs = tf.while_loop(anti_lm_condition, anti_lm_body, 
                                    [concat_base, n_grams_tf, beam_pad, initial_beam_step, n_beams, time], 
                                    shape_invariants=[tf.TensorShape([None, self.vocab_size]), 
                                                      n_grams_tf.get_shape(), beam_pad.get_shape(), 
                                                      initial_beam_step.get_shape(), n_beams.get_shape(), time.get_shape()])

                    anti_beam_probs = anti_lm_outputs[0][1:]
                    
                    score = score - self.anti_lm_weight*tf.to_float(anti_beam_probs)

                    return score

                length_penality_ = tf_helpers._length_penalty(sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)

                score = log_probs/length_penality_

                score = tf.cond(tf.equal(0, time), 
                                            lambda : time_zero_anti_lm(score), lambda: time_not_zero_anti_lm(score))

                return score
            
            def decoder_body(time, outputs, state, inputs, finished, sequence_lengths):       
                
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
  
                def score_total_probs(total_probs):
                    return total_probs
        
                #Here we correct the total_score by the anti-lm prob(T)

                scores = tf.cond(time < self.anti_lm_max_step, 
                                 true_fn=lambda: get_scores_lm(log_probs=total_probs,
                                 sequence_lengths=new_prediction_lengths, length_penalty_weight=self.length_penalty, time=time), 
                                 false_fn=lambda: score_total_probs(total_probs) )
                
                scores_shape = tf.shape(scores)
                
                #Consider only 1 beam at the first step, as we are simply looking for the beam_width number of best
                #tokens to begin with
                scores_flat = tf.cond(time > 0, lambda: tf.reshape(scores, [self.batch_size, -1]), lambda: scores[:, 0])
     
                num_available_beam = tf.cond(time > 0, lambda: tf.reduce_prod(scores_shape[1:]), lambda: tf.reduce_prod(scores_shape[2:]))

                # Pick the next beams according to the specified successors function
                #At this state we have already calculated all the possible combinations of the previous step beam + current beam
                #Thus we are evaluating the top N of sum of P(Current|Previous) + logP(Previous)
                
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
               
                #We have to transform back into beamdecoder class before passing back 
                #as the input and output of the while loop must have same struct
            
                beam_search_state = tf.contrib.seq2seq.BeamSearchDecoderState(
                             cell_state=next_cell_state,
                             log_probs=next_beam_probs,
                             lengths=next_prediction_len,
                             finished=next_finished)

                beam_search_output = tf.contrib.seq2seq.BeamSearchDecoderOutput(
                              scores=next_beam_scores,
                              predicted_ids=next_word_ids,
                              parent_ids=next_beam_ids)
                
                #################################################################
                
                #Here we define the returns
                decoder_finished = beam_search_state.finished
                sample_ids = beam_search_output.predicted_ids
                
                #This part is most important, look up the new seq terms and take their embedding
                #We pass that onto the decoder cell network in the next loop iteration

                next_inputs = tf.cond(tf.reduce_all(decoder_finished), lambda: self.inference_decoder._start_inputs,
                              lambda: self.inference_decoder._embedding_fn(sample_ids))
                ##################################################################

                outputs = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs, beam_search_output)
                
                next_finished = tf.logical_or(decoder_finished, finished)
                
                if maximum_iterations is not None:
                    next_finished = tf.logical_or(
                        next_finished, time + 1 >= maximum_iterations)
                
                next_sequence_lengths = tf.where(
                  tf.logical_and(tf.logical_not(finished), next_finished),
                      tf.fill(tf.shape(sequence_lengths), time + 1),
                  sequence_lengths)
                
                return time+1, outputs, beam_search_state, next_inputs, next_finished, next_sequence_lengths
                
            #Initialize the decoder
                            
            self.time = 0
            
            self.finished, self.first_inputs, self.initial_state = self.inference_decoder.initialize()
            
            initial_time = tf.constant(self.time)
            initial_state = self.initial_state
            initial_inputs = self.first_inputs
            
            if self.max_decode_step is not None:
                maximum_iterations = tf.convert_to_tensor(self.max_decode_step, dtype=tf.int32, name='maximum_iterations')
                initial_finished = tf.logical_or(initial_finished, 0 >= maximum_iterations)

            else:
                maximum_iterations = None
                initial_finished = self.finished
            
            initial_outputs_ta = nest.map_structure(_create_ta, self.inference_decoder.output_size,
                                                    self.inference_decoder.output_dtype)
            
            initial_sequence_lengths = tf.zeros_like(initial_finished, dtype=tf.int32)

            self.decode_loop = tf.while_loop(decoder_condition,
                                        decoder_body,
                                        loop_vars=[initial_time, initial_outputs_ta, initial_state, initial_inputs, 
                                                   initial_finished, initial_sequence_lengths],
                                        parallel_iterations=1,
                                        swap_memory=False)
            
            self.final_outputs_ta = self.decode_loop[1]
            self.final_state = self.decode_loop[2]
            self.final_inputs = self.decode_loop[3]
            
            self.final_outputs = nest.map_structure(lambda ta: ta.stack(), self.final_outputs_ta)
            self.final_sequence_lengths = self.decode_loop[5]
            
            self.final_outputs, self.final_state = self.inference_decoder.finalize(self.final_outputs, 
                                                                              self.final_state, self.final_sequence_lengths)
            
            self.decoder_pred_decode = tf.identity(self.final_outputs.predicted_ids, name='decoder_pred_decode')

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
       
        self.training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_train_inputs_embedded,
                                   sequence_length=self.decoder_train_length,
                                   time_major=False,
                                   name='training_helper')
        
        self.training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                           helper=self.training_helper,
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
            
        #Create handles for encoder
        if self.mode == 'infer':
            
            self.encoder_inputs = tf.placeholder(shape=(None, None),
                        dtype=tf.int32, name='encoder_inputs')

            self.encoder_inputs_length = tf.placeholder(shape=(None,),
                        dtype=tf.int32, name='encoder_inputs_length')
            
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        self.output_layer = Dense(self.vocab_size, name='output_projection')        

        if self.mode == 'train':

            #Make EOS and PAD matrices to concatenate with targets
                
            EOS_SLICE = tf.ones([self.batch_size, 1], dtype=tf.int32) * data_formatting.EOS
            PAD_SLICE = tf.ones([self.batch_size, 1], dtype=tf.int32) * data_formatting.PAD

            #Adding EOS to the beginning of the decoder targets
            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=1, name='decoder_train_inputs_concat')

            self.decoder_train_length = self.decoder_targets_length + 1
            
            self.decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=1, name='decoder_train_targets')
           
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
