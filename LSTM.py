from random import randint
from unittest import main
from numpy import array
from numpy import argmax
from numpy import array_equal
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
from functools import reduce
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def read_file(file):
    f = open(file, 'r')
    return f.read().strip().split('##')

def get_unique_char(str):
    return list(set(reduce(lambda x,y :x+y , [list(set(i)) for i in str])))

    # return output
# prepare data for the LSTM
def get_dataset(n_in, n_out):
    input_data = read_file(n_in)
    output_data = read_file(n_out)
    output_data_input = [f'<sos> {line}' for line in output_data]
    output_data_output = [f'{line} <eos>' for line in output_data]
    input_chars = get_unique_char(input_data)
    target_chars = get_unique_char(output_data)
    max_encoder_seq_len = max([len(text) for text in input_data])
    max_decoder_seq_len = max([len(text) for text in output_data_input])
    return input_data, output_data_input, output_data_output, sorted(input_chars), sorted(target_chars), max_encoder_seq_len, max_decoder_seq_len



# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

def translate_sentence(input_seq, encoder_model, decoder_model, n_features, idx2word_target, n_steps):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((np.shape(states_value)[0], np.shape(states_value)[1], n_features))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(n_steps):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]
    
if __name__=="__main__":
    # generate training dataset
    input_data, output_data_input,\
        output_data_output, input_chars, target_chars,\
            max_encoder_seq_len,\
                max_decoder_seq_len = get_dataset('input_data.txt', 'output_data.txt')

    input_tokenizer = Tokenizer(num_words=max_encoder_seq_len, filters='')
    input_tokenizer.fit_on_texts(input_data)
    input_integer_seq = input_tokenizer.texts_to_sequences(input_data)
    max_input_len = max(len(sen) for sen in input_integer_seq)
    encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
    word2idx_inputs = input_tokenizer.word_index

    output_tokenizer = Tokenizer(num_words=max_decoder_seq_len, filters='')
    output_tokenizer.fit_on_texts(output_data_output + output_data_input)
    output_integer_seq = output_tokenizer.texts_to_sequences(output_data_output)
    output_input_integer_seq = output_tokenizer.texts_to_sequences(output_data_input)
    max_out_len = max(len(sen) for sen in output_integer_seq)
    decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
    decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

    word2idx_outputs = output_tokenizer.word_index
    num_words_output = len(word2idx_outputs) + 1

    idx2word_input = {v:k for k, v in word2idx_inputs.items()}
    idx2word_target = {v:k for k, v in word2idx_outputs.items()}
    # configure problem
    n_features = 50 + 1
    x_encoder = to_categorical(encoder_input_sequences, n_features)
    x_decoder = to_categorical(decoder_input_sequences, n_features)
    y = to_categorical(decoder_output_sequences, n_features)
    # define model
    train, infenc, infdec = define_models(n_features, n_features, 128)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    train.fit([x_encoder, x_decoder], y, epochs=150, verbose=2)
    # evaluate LSTM
    final_predictions = translate_sentence(x_encoder, infenc, infdec, n_features, idx2word_target, max_decoder_seq_len)
    print(final_predictions)

    print("completed")