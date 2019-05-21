import tensorflow as tf


def mask_to_start(score, start, score_mask_value=-1e30):
    score_mask = tf.cast(tf.ones_like(start) - tf.cumsum(start, axis=-1), tf.float32)
    return score + score_mask * score_mask_value


def mask_to_topk(score, topk_ind, c_maxlen, score_mask_value=-1e30):
    score_mask = tf.reduce_sum(tf.one_hot(topk_ind, depth=c_maxlen), axis=-2)  # [bs, topk]->[bs, topk, c_len]->[bs, c_len]
    score_mask = tf.cast(tf.ones_like(score_mask) - score_mask, tf.float32)
    return score + score_mask * score_mask_value


def get_tf_f1(y_pred, y_true):
    y_true = tf.cast(y_true, tf.float32)
    y_union = tf.clip_by_value(y_pred + y_true, 0, 1)  # [bs, c_maxlen]
    y_diff = tf.abs(y_pred - y_true)  # [bs, c_maxlen]
    num_same = tf.cast(tf.reduce_sum(y_union, axis=-1) - tf.reduce_sum(y_diff, axis=-1), tf.float32)  # [bs,] ## 동일한 개수
    y_precision = num_same / (tf.cast(tf.reduce_sum(y_pred, axis=-1), tf.float32) + 1e-7)  # [bs,] ## precision
    y_recall = num_same / (tf.cast(tf.reduce_sum(y_true, axis=-1), tf.float32) + 1e-7)  # [bs,] ## recall
    y_f1 = (2.0 * y_precision * y_recall) / (tf.cast(y_precision + y_recall, tf.float32) + 1e-7)  # [bs,] ## 2 * precision * recall / (precision + recall)
    return tf.clip_by_value(y_f1, 0, 1)


def rl_loss(logits_start, logits_end, y_start, y_end, c_maxlen, rl_loss_type = 'topk_DCRL', topk=3):
    assert rl_loss_type == 'DCRL' or rl_loss_type == 'SCST' or rl_loss_type == 'topk_DCRL'
    # get ground truth prediction
    # s:[0,1,0,0,0], e:[0,0,0,1,0]->[0,1,1,1,1]-[0,0,0,1,1]->[0,1,1,0,0]+e:[0,0,0,1,0]->pred:[0,1,1,1,0]
    y_start_cumsum = tf.cumsum(y_start, axis=-1) ## http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn/api_docs/python/tf/cumsum.html
    y_end_cumsum = tf.cumsum(y_end, axis=-1)
    ground_truth = y_start_cumsum - y_end_cumsum + y_end  # [bs, c_maxlen] ## 정답 구간

    # get greedy prediction
    greedy_start = tf.one_hot(tf.argmax(logits_start, axis=-1), c_maxlen,
                              axis=-1)  # [bs, c_maxlen]->[bs,]->[bs, c_maxlen] ## greedy -> argmax: 시작 점
    masked_logits_end = mask_to_start(logits_end, greedy_start)
    greedy_end = tf.one_hot(tf.argmax(masked_logits_end, axis=-1), c_maxlen, axis=-1) ## greedy -> argmax: 끝 점
    greedy_start_cumsum = tf.cumsum(greedy_start, axis=-1)
    greedy_end_cumsum = tf.cumsum(greedy_end, axis=-1)
    greedy_prediction = greedy_start_cumsum - greedy_end_cumsum + greedy_end  # [bs, c_maxlen] ## 예측 구간
    # get greedy f1
    greedy_f1 = get_tf_f1(greedy_prediction, ground_truth) ## greedy f1

    # get sampled prediction (use tf.multinomial)
    if rl_loss_type == 'topk_DCRL':
        start_topk_ind = tf.nn.top_k(logits_start, topk).indices  # [bs, topk_size]
        masked_logits_start = mask_to_topk(logits_start, start_topk_ind, c_maxlen)
    else:
        masked_logits_start = logits_start
    sampled_start_ind = tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(masked_logits_start)), 1),
                                   axis=-1)  # [bs, c_maxlen]->[bs, 1]->[bs,] ## softmax를 이용한 확률 예측
    sampled_start = tf.one_hot(sampled_start_ind, c_maxlen, axis=-1)  # [bs, c_maxlen]->[bs,]->[bs, c_maxlen]
    masked_logits_end = mask_to_start(logits_end, sampled_start)
    if rl_loss_type == 'topk_DCRL':
        end_topk_ind = tf.nn.top_k(masked_logits_end, topk).indices  # [bs, topk_size]
        masked_logits_end = mask_to_topk(masked_logits_end, end_topk_ind, c_maxlen)
    sampled_end_ind = tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(masked_logits_end)), 1), axis=-1) ## softmax를 이용한 확률 예측
    sampled_end = tf.one_hot(sampled_end_ind, c_maxlen, axis=-1)
    sampled_start_cumsum = tf.cumsum(sampled_start, axis=-1)
    sampled_end_cumsum = tf.cumsum(sampled_end, axis=-1)
    sampled_prediction = sampled_start_cumsum - sampled_end_cumsum + sampled_end  # [bs, c_maxlen]
    # get sampled f1
    sampled_f1 = get_tf_f1(sampled_prediction, ground_truth) ## sample f1 계산

    reward = tf.stop_gradient(sampled_f1 - greedy_f1)  # (sampled - baseline) ## https://www.tensorflow.org/api_docs/python/tf/stop_gradient
    sampled_start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_start, labels=sampled_start)
    sampled_end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_end, labels=sampled_end)

    if rl_loss_type == 'DCRL' or rl_loss_type == 'topk_DCRL':
        reward = tf.clip_by_value(reward, 0., 1e7)
        reward_greedy = tf.clip_by_value(tf.stop_gradient(greedy_f1 - sampled_f1), 0., 1e7)
        greedy_start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_start, labels=greedy_start)
        greedy_end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_end, labels=greedy_end)
        return tf.reduce_mean(reward * (sampled_start_loss + sampled_end_loss) + reward_greedy * (
                greedy_start_loss + greedy_end_loss)), sampled_f1, greedy_f1
    elif rl_loss_type == 'SCST':
        return tf.reduce_mean(reward * (sampled_start_loss + sampled_end_loss)), sampled_f1, greedy_f1
