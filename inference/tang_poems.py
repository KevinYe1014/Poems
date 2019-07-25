import collections
import os, sys, numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.poems import process_poems, generate_batch
import heapq

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64, 'batch size. ')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate. ')
# set this to 'main.py' relative path
flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/poems'), 'checkpoint dir')
flags.DEFINE_string('file_path', os.path.abspath('./dataset/data/poems.txt'), 'data file')
flags.DEFINE_string('model_prefix', 'poems', 'model save prefix')
flags.DEFINE_integer('epochs', 50, 'train how many epochs. ')


FLAGS = flags.FLAGS

start_token = 'G'
end_token = 'E'

def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None] )
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets,
                           vocab_size=len(vocabularies), rnn_size=128, num_layers=2, batch_size=64,
                           learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver()
    '''
    tf.group(tensor1, tensor2)：
    tensor1和tensor2是操作，用于操作集合起来。比如：
    generator_train_op = tf.train.AdamOptimizer(g_loss, ...)
    discriminator_train_op = tf.train.AdamOptimizer(d_loss,...)
    train_ops = tf.groups(generator_train_op ,discriminator_train_op)
    with tf.Session() as sess:
        sess.run(train_ops) 
    一旦运行了train_ops,那么里面的generator_train_op和discriminator_train_op都将被调用
    这里注意的是：tf.group()返回的是个操作，而不是值，如果你想tensor1和tensor2填充Variable 那么返回是None
    如果真想返回值，那么可以用tf.tuple()
    全局变量：用来初始化计算图中的全局的变量，全局变量是指创建的变量在tf.GraphKeys.GLOBAL_VARIABLES中，
    在使用Variable创建变量时默认是collections默认是tf.GraphKeys.GLOBAL_VARIABLES
    局部变量：初始化计算图中所有的局部变量，局部变量是指创建的变量在tf.GraphKeys.LOCAL_VARIABLES中
    a = tf.Variable(1,name="a",collections=[tf.GraphKeys.LOCAL_VARIABLES])
    备注：在使用saver的时候，局部变量是不存在在模型文件中的
    '''
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] restore from the checkpoint %s' % checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training ... ')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([end_points['total_loss'], end_points['last_state'],
                                           end_points['train_op']], feed_dict={input_data: batches_inputs[n], output_targets: batches_inputs[n]})
                    n += 1

                if epoch % 1 == 0:
                    saver.save(sess, './model/', global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now ...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch %d' % epoch )

def gen_poem(begin_word):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)
    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, './model/-24')

        x = np.array([list(map(word_int_map.get, start_token))])
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        while word != end_token and word != start_token and word != '':
            print('running ... ')
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
            # word = vocabularies[np.argmax(predict)]
        return poem


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs) - 1:
        sample = len(vocabs) - 1
    return vocabs[sample]



def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' : # and len(s) > 10
            print(s + '。')

def main(is_train):
    if is_train:
        print('[INFO] train tang poem ...')
        run_training()
    else:
        print('[INFO] write tang poem ...')
        begin_word = input('输入起始字：')
        # begin_word = '我'
        poem2 = gen_poem(begin_word)
        pretty_print_poem(poem2)


if __name__ == '__main__':
    tf.app.run()


