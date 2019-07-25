import tensorflow as tf
import numpy as np

def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128,
              num_layers=2, batch_size=64, learning_rate=0.01):
    """
    construct rnn seq2seq model
    :param model:
    :param input_data:
    :param output_data:
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param learning_rate:
    :return:
    """
    end_points = {}

    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    '''
    tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True): 
    n_hidden表示神经元的个数，forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。
    如果等于0，就都忘记。state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示。
    这个里面存在一个状态初始化函数，就是zero_state（batch_size，dtype）两个参数。batch_size就是输入样
    本批次的数目，dtype就是数据类型。

    '''

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    '''
    这个地方，注意rnn_size是影层神经元的个数，相当于全连接层中的神经元的个数，然后num_layers是
    有多少个这样的影层，因为这里是深度lstm模型，有多个同样的lstm单元个数
    '''
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device('/cpu:0'):
        # embedding (6111, 128) 里面value是按照正态分布取值的数
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        # input_data (64, None) none可能是10,14等  比如 [[2, 45, 3, 5, 12, 23, 21, 7, 8, 63], ...... ]

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    # state_is_tuple=True，如果为True，则接受和返回的状态是c_state和m_state的2-tuple；如果为False，则他们沿着列轴连接。
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size + 1]

    if output_data is not None:
        # output_data must be one-hot encode 输出 (64*None, 61111)  tf.reshape(output_data, [-1]) 直接拉平了 相当于flatten
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size + 1]

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size + 1]
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points
















