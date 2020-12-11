
The encoder-decoder model provides a pattern for using recurrent neural networks to address challenging
sequence-to-sequence prediction problems such as machine translation. Encoder-decoder models can be
developed in the Keras Python deep learning library and an example of a neural machine translation system
developed with this model has been described on the Keras blog, with sample code distributed with the Keras project.
This project provides the basis for developing encoder-decoder LSTM models with sequence-to-sequence
prediction problem.

Testing:
python -m unittest unit_test.py

Required:
Python 2 or 3. Keras (2.0 or higher) installed with either the TensorFlow or Theano backend,
 scikit-learn, Pandas, NumPy, and Matplotlib installed.
 keras.preprocessing.text for Tokenizer
 keras.preprocessing.sequence for pad_sequences

Resources:
https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/
https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
https://realpython.com/python-testing/#unit-tests-vs-integration-tests
