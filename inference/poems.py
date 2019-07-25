import numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.poems import process_poems, generate_batch

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64, 'batch size = ?')
flags.DEFINE_float('learning', 0.01, 'learning rate')
flags.DEFINE_string('checkpoint_dir', './model/', 'checkpoint dir')
flags.DEFINE_string('file_path', './data/.txt', 'file path')
flags.DEFINE_integer('epochs', 50, 'train epochs')
FLAGS = flags.FLAGS

start_token = 'G'
end_token = 'E'

def run_training():
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batch_inputs, batch_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(vocabularies),
                           rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning )

def main(is_train):
    if is_train:
        print('training ... ')
        run_training()
    else:
        print('test ... ')
        begin_word = input('word: ')

if __name__ == '__main__':
    tf.app.run()
