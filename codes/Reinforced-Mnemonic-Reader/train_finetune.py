import numpy as np
import pandas as pd
import RMR_modelV6
import tensorflow as tf
import json
import os
import util
import time
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def training_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)
    index = np.arange(data[0].shape[0])
    np.random.shuffle(index)
    for i, d in enumerate(data):
        if len(d.shape) > 1:
            data[i] = data[i][index, ::]
        else:
            data[i] = data[i][index]
    return data


def next_batch(data, batch_size, iteration):
    data_temp = []
    start_index = iteration * batch_size
    end_index = (iteration + 1) * batch_size
    for i, d in enumerate(data):
        data_temp.append(data[i][start_index: end_index, ::])
    return data_temp


def cal_ETA(t_start, i, n_batch):
    t_temp = time.time()
    t_avg = float(int(t_temp) - int(t_start)) / float(i + 1)
    if n_batch - i - 1 > 0:
        return int((n_batch - i - 1) * t_avg)
    else:
        return int(t_temp) - int(t_start)


# load trainset
context_word = np.load('../QANet_tf/dataset1.0/train_contw_input.npy').astype(np.int32)
question_word = np.load('../QANet_tf/dataset1.0/train_quesw_input.npy').astype(np.int32)
context_char = np.load('../QANet_tf/dataset1.0/train_contc_input.npy').astype(np.int32)
question_char = np.load('../QANet_tf/dataset1.0/train_quesc_input.npy').astype(np.int32)
start_label = np.load('../QANet_tf/dataset1.0/train_y_start.npy').astype(np.int32)
end_label = np.load('../QANet_tf/dataset1.0/train_y_end.npy').astype(np.int32)
context_string = np.load('../QANet_tf/dataset1.0/train_contw_strings.npy')
ques_string = np.load('../QANet_tf/dataset1.0/train_quesw_strings.npy')

# load valset
val_context_word = np.load('../QANet_tf/dataset1.0/dev_contw_input.npy').astype(np.int32)
val_question_word = np.load('../QANet_tf/dataset1.0/dev_quesw_input.npy').astype(np.int32)
val_context_char = np.load('../QANet_tf/dataset1.0/dev_contc_input.npy').astype(np.int32)
val_question_char = np.load('../QANet_tf/dataset1.0/dev_quesc_input.npy').astype(np.int32)
val_start_label = np.load('../QANet_tf/dataset1.0/dev_y_start.npy').astype(np.int32)
val_end_label = np.load('../QANet_tf/dataset1.0/dev_y_end.npy').astype(np.int32)
val_qid = np.load('../QANet_tf/dataset1.0/dev_qid.npy').astype(np.int32)
val_context_string = np.load('../QANet_tf/dataset1.0/dev_contw_strings.npy')
val_ques_string = np.load('../QANet_tf/dataset1.0/dev_quesw_strings.npy')

with open('../QANet_tf/dataset1.0/test_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
word_mat = np.load('../QANet_tf/dataset1.0/word_emb_mat.npy')
char_mat = np.load('../QANet_tf/dataset1.0/char_emb_mat.npy')

train_set = [context_word, question_word, context_char, question_char, context_string, ques_string, start_label,
             end_label]
val_set = [val_context_word, val_question_word, val_context_char, val_question_char, val_context_string,
           val_ques_string, val_start_label, val_end_label]

config = {
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': 50,
    'filters': 100,
    'dropout': 0.3,
    'l2_norm': 3e-7,
    'decay': 0.9999,
    'learning_rate': 1e-4,
    'grad_clip': 5.0,
    'batch_size': 32,
    'epoch': 20,
    'per_steps': 500,
    'init_lambda': 3.0,
    'rl_loss_type': 'DCRL', # ['SCTC', 'DCRL', 'topk_DCRL', None]
    'origin_path': 'RMRV0',
    'path': 'RMRV0_f'
}

model = RMR_modelV6.Model(config, word_mat=word_mat, char_mat=char_mat, elmo_path="../QANet_tf/tfhub_elmo")
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True

best_f1 = 0
best_em = 0
f1s = []
ems = []

with tf.Session(config=sess_config) as sess:
    if not os.path.exists(os.path.join('model', config['path'])):
        os.mkdir(os.path.join('model', config['path']))
    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore(include=['Input_Embedding_Layer',
                                                                  'Iterative_Reattention_Aligner',
                                                                  'Answer_Pointer'])
    saver = tf.train.Saver(variables_to_restore)
    if os.path.exists(os.path.join('model',config['origin_path'],'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join('model',config['origin_path'])))
    n_batch = context_word.shape[0] // config['batch_size']
    n_batch_val = val_context_word.shape[0] // config['batch_size']

    # during the finetune with rl_loss we validate the result per 500 steps
    for epoch in range(config['epoch']):
        train_set = training_shuffle(train_set)
        last_train_str = "\r"
        # training step
        sum_loss = 0
        sum_rl_loss = 0
        for i in range(n_batch):
            contw_input, quesw_input, contc_input, quesc_input, contw_string, quesw_string, y_start, y_end \
                = next_batch(train_set, config['batch_size'], i)
            loss_value, rl_loss_value, theta_a, theta_b, sampled_f1, greedy_f1, _ = sess.run([model.loss, model.rl_loss, model.theta_a, model.theta_b,
                                                                       model.sampled_f1, model.greedy_f1, model.train_op],
                                     feed_dict={model.contw_input_: contw_input, model.quesw_input_: quesw_input,
                                                model.contc_input_: contc_input, model.quesc_input_: quesc_input,
                                                model.contw_strings: contw_string, model.quesw_strings: quesw_string,
                                                model.y_start_: y_start, model.y_end_: y_end,
                                                model.dropout: config['dropout']})
            sum_loss += loss_value
            sum_rl_loss += rl_loss_value
            last_train_str = "\r[epoch:%d/%d, steps:%d/%d] loss:%.4f rl_loss:%.4f" % (
                epoch + 1, config['epoch'], i + 1, n_batch, sum_loss/(i+1), rl_loss_value)
            print(last_train_str, end='      ', flush=True)
            # print('sf1:',sampled_f1)
            # print('gf1:',greedy_f1)
            if (i+1)%config['per_steps']==0 or i+1==n_batch:
                # validating step
                sum_loss_val = 0
                sum_rl_loss_val = 0
                y1s = []
                y2s = []
                last_val_str = "\r"
                for i in range(n_batch_val):
                    contw_input, quesw_input, contc_input, quesc_input, contw_string, quesw_string, y_start, y_end \
                        = next_batch(val_set, config['batch_size'], i)
                    loss_value, rl_loss_value, y1, y2 = sess.run([model.loss, model.rl_loss, model.output1, model.output2],
                                                  feed_dict={model.contw_input_: contw_input,
                                                             model.quesw_input_: quesw_input,
                                                             model.contc_input_: contc_input,
                                                             model.quesc_input_: quesc_input,
                                                             model.contw_strings: contw_string,
                                                             model.quesw_strings: quesw_string,
                                                             model.y_start_: y_start, model.y_end_: y_end})
                    y1s.append(y1)
                    y2s.append(y2)
                    sum_loss_val += loss_value
                    sum_rl_loss_val += rl_loss_value
                    last_val_str = last_train_str + " [validate:%d/%d] loss:%.4f rl_loss:%.4f" % (
                        i + 1, n_batch_val, sum_loss_val / (i + 1), rl_loss_value)
                    print(last_val_str, end='      ', flush=True)
                y1s = np.concatenate(y1s)
                y2s = np.concatenate(y2s)
                answer_dict, _, noanswer_num = util.convert_tokens(eval_file, val_qid.tolist(), y1s.tolist(),
                                                                   y2s.tolist(),
                                                                   data_type=1)
                metrics = util.evaluate(eval_file, answer_dict)
                ems.append(metrics['exact_match'])
                f1s.append(metrics['f1'])

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    saver.save(sess, os.path.join('model', config['path'], 'model.ckpt'),
                               global_step=(epoch + 1) * n_batch)

                print(last_val_str,
                      " -EM: %.2f%%, -F1: %.2f%% -Noanswer: %d" % (metrics['exact_match'], metrics['f1'], noanswer_num),
                      end=' ', flush=True)
                print('\n')

                result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
                result.to_csv('log/result_' + config['path'] + '.csv', index=None)

        saver.save(sess, os.path.join('model', config['path'], 'model.ckpt'), global_step=config['epoch'] * n_batch)