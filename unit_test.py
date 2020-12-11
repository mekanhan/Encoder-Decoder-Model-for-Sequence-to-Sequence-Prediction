###Unit Testing
import unittest
from LSTM import read_file, get_unique_char, get_dataset, define_models, one_hot_decode

class Test(unittest.TestCase):

    def test_read_files(self):
        actual = read_file("test_file.txt")
        expected = ["A\n", "\nB"]
        self.assertEqual(actual, expected)

    def test_get_unique_char(self):
        actual = get_unique_char(['ABCABCABC'])
        expected = ['A', 'B', 'C']
        self.assertCountEqual(actual, expected)
    
    def test_get_dataset(self):
        input_data, output_data_input,\
        output_data_output, input_chars, target_chars,\
            max_encoder_seq_len,\
                max_decoder_seq_len = get_dataset('test_file.txt', 'test_file.txt')
        expected_in = ["A\n", "\nB"]
        expected_dec_in = ["<sos> A\n", "<sos> \nB"]
        expected_out = ["A\n <eos>", "\nB <eos>"]
        expected_max_len = 2
        expected_chars = ['A', 'B', '\n']
        with self.subTest("testing input"):
            self.assertEqual(input_data, expected_in)
        with self.subTest("testing dec input"):
            self.assertEqual(output_data_input, expected_dec_in)
        with self.subTest("testing dec output"):
            self.assertEqual(output_data_output, expected_out)
        with self.subTest("Testing unique input chars"):
            self.assertCountEqual(input_chars, expected_chars)

    def test_define_models(self):
        model, enc,dec = define_models(3,3,32)
        with self.subTest():
            self.assertEqual(model.inputs[0].shape[-1], 3)
        with self.subTest():
            self.assertEqual(enc.outputs[0].shape[-1], 32)
        with self.subTest():
            self.assertEqual(dec.inputs[0].shape[-1], 3)
        with self.subTest():
            self.assertEqual(dec.outputs[0].shape[-1], 3)

    def test_one_hot_decode(self):
        actual = one_hot_decode([[0,0,1], [0,1,0], [1,0,0]])
        expected = [2, 1, 0]
        self.assertEqual(actual, expected)

##End Unit Testing