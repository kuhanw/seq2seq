# seq2seq-learning

#Introduction

This my implementation of a seq2seq Neural translation model using the Tensorflow API. There are many tutorials and examples on the internet demonstrating how to implement a seq2seq translation model in Tensorflow, including from Google (https://github.com/tensorflow/nmt), I studied many of them and recreated my own as a way to teach myself the core functionality of Tensorflow and to internalize my understanding of many of the underlying ideas behind seq2seq and deep learning in general.

In order to run the code you need: Python 3.5, Tensorflow 1.3. Pandas 0.18

The training code can be launched from seq2seq_training.py, for example: 

python seq2seq_training.py -s ../../efs/chkpt -cells 256 -n_layers 3 -n_embedding 256 -dropout 0.8 -beam_length 10 -minibatch_size 128 -data data_file.pkl -vocab vocabulary.pkl

where data_file.pkl andd vocabulary.pkl are Python 3 compatible pickles consisting of:
  
  - data_file.pkl, rows of encoder, decoder inputs in numerical representation as a pandas dataframe.
  - vocabulary.pkl, a dictionary of key:value mappings between string tokens and integers. 
  
representative examples can be found in this github repository under: https://github.com/kuhanw/processed_data.

Once you have a checkpointed model, at any time you can perform inference (i.e. return a response with only the encoder input) by
executing seq2seq_training.py:

python seq2seq_inference.py -r checkpoint_file -cells 256 -n_layers 3 -n_embedding 256 -beam_length 10 -vocab vocabulary.pkl	-freeze frozen_model_path -input encoder_input 

where

  - checkpoint_file is the checkpoint to restore from
  - encoder_input is the string input to the model
  - frozen_model_path is a optional param if you wish to create a frozen instance of the model for serving as a API.
  
A simple API serving the model will be at a later date.

Some things to do when I get around to it:

  - Implement a bidirectional encoder
  - Implement anti-language model to increase the diversity of the decoder inference responses (Marsan Ma has a version implemented here: https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm)
  - Clean up API code to use frozen models instead of restoring from checkpoint
  - Clean up notebooks
