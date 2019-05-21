import tensorflow as tf
from tensor2tensor.layers.common_layers import conv1d
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.contrib.keras import backend
from tensorflow.contrib.layers import variance_scaling_initializer, l2_regularizer

initializer = lambda: variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32)
initializer_relu = lambda: variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)
regularizer = l2_regularizer(scale=3e-7)


# cudnnLSTM
def BiLSTM(x, filters, dropout=0.0, name='BiLSTM', layers=1, return_state=False):
    cudnn_lstm = CudnnLSTM(layers, filters, direction='bidirectional', name=name)
    if type(x) == list:
        assert len(x) == 2
        x1, x2 = x
        # cudnn compatibility: time first, batch second
        x1 = tf.transpose(x1, [1, 0, 2])
        x2 = tf.transpose(x2, [1, 0, 2])
        x1, x1_state = cudnn_lstm(x1)  # state:[2, bs, dim]
        x2, x2_state = cudnn_lstm(x2)
        x1 = tf.transpose(x1, [1, 0, 2])
        x2 = tf.transpose(x2, [1, 0, 2])
        x1_state = tf.concat(tf.unstack(x1_state[0], axis=0), axis=-1)
        x2_state = tf.concat(tf.unstack(x2_state[0], axis=0), axis=-1)
        if return_state:
            return tf.nn.dropout(x1_state, 1 - dropout), tf.nn.dropout(x2_state, 1 - dropout)
        else:
            return tf.nn.dropout(x1, 1 - dropout), tf.nn.dropout(x2, 1 - dropout)
    else:
        # cudnn compatibility: time first, batch second
        x = tf.transpose(x, [1, 0, 2])
        x, x_state = cudnn_lstm(x)
        if return_state:
            x_state = tf.concat(tf.unstack(x_state[0], axis=0), axis=-1)
            return tf.nn.dropout(x_state, 1 - dropout)
        else:
            x = tf.transpose(x, [1, 0, 2])
            return tf.nn.dropout(x, 1 - dropout)


def exp_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def align_block(u, v, c_mask, q_mask, Lambda, filters=128, E_0=None, B_0=None, Z_0=None, dropout=0.0):
    with tf.variable_scope("Interactive_Alignment"):
        # attention
        u_ = tf.nn.relu(conv1d(u, filters, 1, name="Wu"))  # [bs, len_c, dim]
        v_ = tf.nn.relu(conv1d(v, filters, 1, name="Wv"))  # [bs, len_q, dim]
        E = tf.matmul(v_, u_, transpose_b=True)  # [bs, len_q, len_c] ## relu(WuU)*relu(WvV)
        if E_0 is not None: ## block 2, 3
            E += (Lambda * E_0)
        E_ = tf.nn.softmax(exp_mask(E, tf.expand_dims(q_mask, axis=-1)), axis=1)  # [bs, len_q, len_c]
        v_E = tf.matmul(E_, v, transpose_a=True)  # [bs, len_c, dim] ## v~ = V * softmax(E)

        # fusion
        uv = tf.concat([u, v_E, u * v_E, u - v_E], axis=-1) ## x; y; x * y; x - y
        x = tf.nn.relu(conv1d(uv, filters, 1, name='Wr')) ## x~ = relu(Wr[x; y; x * y; x - y])
        g = tf.nn.sigmoid(conv1d(uv, filters, 1, name='Wg')) ## g = sigmoid(Wg[x; y; x * y; x - y])
        h = g * x + (1 - g) * u  # [bs, len_c, dim] ## o = g * x~ + (1 - g) * x

    with tf.variable_scope("Self_Alignment"):
        # attention
        h_1 = tf.nn.relu(conv1d(h, filters, 1, name='Wh1')) ## h1 = relu(Wh1 * h)
        h_2 = tf.nn.relu(conv1d(h, filters, 1, name='Wh2')) ## h2 = relu(Wh2 * h)
        B = tf.matmul(h_2, h_1, transpose_b=True)  # [bs, len_c, len_c] ## seftattention
        if B_0 is not None: ## block 2, 3
            B += (Lambda * B_0)
        B_ = tf.nn.softmax(exp_mask(B, tf.expand_dims(c_mask, axis=-1)), axis=1)  # [bs, len_c, len_c]
        h_B = tf.matmul(B_, h, transpose_a=True) ## H~ = H * softmax(B)

        # fusion
        hh = tf.concat([h, h_B, h * h_B, h - h_B], axis=-1)
        x = tf.nn.relu(conv1d(hh, filters, 1, name='Wr'))
        g = tf.nn.sigmoid(conv1d(hh, filters, 1, name='Wg'))
        Z = g * x + (1 - g) * h  # [bs, len_c, dim]

    with tf.variable_scope("Evidence_Collection"):
        if Z_0 is not None: ## block 3
            Z = tf.concat([Z, Z_0[0], Z_0[1]], axis=-1)
        R = BiLSTM(Z, filters // 2, name='bilstm', dropout=dropout)  # [bs, len_c, dim] ## R = BiLSTM(Z)

    # return the E_t, B_t
    E_t = tf.nn.softmax(exp_mask(E, tf.expand_dims(c_mask, axis=1)), axis=-1)  # [bs, len_q, len_c]
    E_t = tf.matmul(E_t, B_) ## ???
    B_t = tf.nn.softmax(exp_mask(B, tf.expand_dims(c_mask, axis=1)), axis=-1)  # [bs, len_c, len_c]
    B_t = tf.matmul(B_t, B_) ## ???

    return R, Z, E_t, B_t


def summary_vector(q_emb, c_maxlen, mask):
    with tf.variable_scope("Question_Summary"):
        alpha = tf.nn.softmax(exp_mask(tf.squeeze(conv1d(q_emb, 1, 1), axis=-1), mask)) ## ai
        s = tf.expand_dims(alpha, axis=-1) * q_emb ## ai * vi
        s = tf.reduce_sum(s, axis=1, keepdims=True)  # [bs, 1, dim]
        s = tf.tile(s, [1, c_maxlen, 1])  # [bs, len_c, dim] ## https://dhhwang89.tistory.com/116
    return s


def start_logits(R, s, mask, filters=128, name='Start_Pointer'):
    with tf.variable_scope(name):
        if R.get_shape()[-1] == s.get_shape()[-1]:
            logits1 = tf.concat([R, s, R * s, R - s], axis=-1) ## R; s; R * s; R - s 
        else:
            logits1 = tf.concat([R, s], axis=-1)
        logits1 = tf.nn.tanh(conv1d(logits1, filters, 1, name='Wt')) ## tanh(W1 * logit)
        logits1 = tf.squeeze(conv1d(logits1, 1, 1, name='Wf'), axis=-1)
        logits1 = exp_mask(logits1, mask) ## exp
    return logits1


def end_logits(R, logits1, s, mask, filters=128, name='End_Pointer'):
    with tf.variable_scope(name):
        l = R * tf.expand_dims(tf.nn.softmax(logits1, axis=-1), axis=-1)  # [bs, len_c, dim] ## R3 * p1
        if s.get_shape()[-1] == l.get_shape()[-1]:
            s_ = tf.concat([s, l, s * l, s - l], axis=-1) ## x; y; x * y; x - y
        else:
            s_ = tf.concat([s, l], axis=-1)
        x = tf.nn.relu(conv1d(s_, filters, 1, name='Wr'))  # [bs, len_c, dim] ## relu(Wr * [x; y; x * y; x - y])
        g = tf.nn.sigmoid(conv1d(s_, filters, 1, name='Wg'))  # [bs, len_c, dim] ## sigmoid(Wg * [x; y; x * y; x - y])
        s_ = g * x + (1 - g) * s  # [bs, len_c, dim] ## g * x~ + (1 - g) * x

        if R.get_shape()[-1] == s_.get_shape()[-1]:
            logits2 = tf.concat([R, s_, R * s_, R - s_], axis=-1) ## R; s~; R * s~; R - s~
        else:
            logits2 = tf.concat([R, s_], axis=-1)
        logits2 = tf.nn.tanh(conv1d(logits2, filters, 1, name='Wt')) ## ## tanh(W2 * logit)
        logits2 = tf.squeeze(conv1d(logits2, 1, 1, name='Wf'), axis=-1)
        logits2 = exp_mask(logits2, mask) ## exp
    return logits2


def ElmoCombineLayer(elmo_feats, name):  # [bs, len, 3, 1024]
    n_lm_layers = int(elmo_feats.get_shape()[2])  # 3
    W = tf.get_variable(
        '{}_ELMo_W'.format(name),
        shape=(n_lm_layers,),
        initializer=tf.zeros_initializer,
        regularizer=regularizer,
        trainable=True,
    )
    normed_weights = tf.split(tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers)  # [1]*3
    # split LM layers
    layers = tf.split(elmo_feats, n_lm_layers, axis=2)  # [bs, len, 1, 1024]*3

    # compute the weighted, normalized LM activations
    pieces = []
    for w, t in zip(normed_weights, layers):
        pieces.append(w * tf.squeeze(t, axis=2))
    sum_pieces = tf.add_n(pieces)

    # scale the weighted sum by gamma
    gamma = tf.get_variable(
        '{}_ELMo_gamma'.format(name),
        shape=(1,),
        initializer=tf.ones_initializer,
        regularizer=None,
        trainable=True,
    )
    return sum_pieces * gamma  # [bs, len, 1024]


def CoveCombineLayer(cove_feats, name):  # [bs, len, 2, 600]
    n_lm_layers = int(cove_feats.get_shape()[2])  # 2
    W = tf.get_variable(
        '{}_Cove_W'.format(name),
        shape=(n_lm_layers,),
        initializer=tf.zeros_initializer,
        regularizer=regularizer,
        trainable=True,
    )
    normed_weights = tf.split(tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers)  # [1]*2
    # split LM layers
    layers = tf.split(cove_feats, n_lm_layers, axis=2)  # [bs, len, 1, 600]*2

    # compute the weighted, normalized LM activations
    pieces = []
    for w, t in zip(normed_weights, layers):
        pieces.append(w * tf.squeeze(t, axis=2))
    sum_pieces = tf.add_n(pieces)

    # scale the weighted sum by gamma
    gamma = tf.get_variable(
        '{}_Cove_gamma'.format(name),
        shape=(1,),
        initializer=tf.ones_initializer,
        regularizer=None,
        trainable=True,
    )
    return sum_pieces * gamma  # [bs, len, 600]


def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0, scope='efficient_trilinear',
                                      bias_initializer=tf.zeros_initializer(), kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        subres0 = tf.tile(backend.dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(backend.dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        subres2 = backend.batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        res += biases
        return res


def ElmoAttention(inputs, c_maxlen, q_maxlen, q_mask, dropout):
    c, q = inputs
    S = optimized_trilinear_for_attention([c, q], c_maxlen, q_maxlen, input_keep_prob=1. - dropout,
                                          scope='elmo_efficient_trilinear')
    q_mask = tf.expand_dims(q_mask, 1)
    S_ = tf.nn.softmax(exp_mask(S, mask=q_mask))
    c2q = tf.matmul(S_, q)
    return tf.concat([c, c2q], axis=-1)


def total_params(exclude=None):
    total_parameters = 0
    if exclude is not None:
        trainable_variables = list(set(tf.trainable_variables()) ^ set(tf.trainable_variables(exclude)))
    else:
        trainable_variables = tf.trainable_variables()
    for variable in trainable_variables:
        shape = variable.get_shape()
        variable_parametes = 1
        try:
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        except:
            print(shape, 'cudnn weights is unknown')
    print("Total number of trainable parameters: {}".format(total_parameters))
