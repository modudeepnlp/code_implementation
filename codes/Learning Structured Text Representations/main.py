from data_structure import DataSet
import tensorflow as tf
import numpy as np
import cPickle
import logging
from models import  StructureModel
import tqdm

# function : load_data, evaluate, run (run부터 살펴보기)

def load_data(config):
    train, dev, test, embeddings, vocab = cPickle.load(open(config.data_file))
    trainset, devset, testset = DataSet(train), DataSet(dev), DataSet(test)
    vocab = dict([(v.index,k) for k,v in vocab.items()])
    trainset.sort()
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    return len(train), train_batches, dev_batches, test_batches, embeddings, vocab

def evaluate(sess, model, test_batches):
    corr_count, all_count = 0, 0
    for ct, batch in test_batches:
        feed_dict = model.get_feed_dict(batch)
        feed_dict[model.t_variables['keep_prob']] = 1
        predictions = sess.run(model.final_output, feed_dict=feed_dict)

        predictions = np.argmax(predictions, 1)
        corr_count += np.sum(predictions == feed_dict[model.t_variables['gold_labels']])
        all_count += len(batch)
    acc_test = 1.0 * corr_count / all_count
    return  acc_test

def run(config):
    import random

    hash = random.getrandbits(32) # 32자리 비트를 랜덤 생성
    # logging : DEBUG < INFO < *WARNING < *ERROR < *CRITICAL (앞의 두 레벨은 따로 설정해야 출력해준다.)
    # 1. 생성, 2. 레벨 설정, 3. (파일)핸들러 설정(내가 로깅한 정보가 (파일로) 출력되는 위치 설정하는 것), 4. 출력 포매팅 설정

    # logger <- ah <- formatter
    logger = logging.getLogger() # 1. 자신만의 특정한 로거 만들기(루트로거 리턴, 기본레벨은 warning)
    logger.setLevel(logging.DEBUG) # 2. DEBUG까지 출력해줘라.

    ah = logging.FileHandler(str(hash)+'.log') # 3. 해당 디렉토리에 파일이 만들어진다.
    ah.setLevel(logging.DEBUG) # 3. 여기에서도 레벨을 정해준다. 
    
    formatter = logging.Formatter('%(asctime)s - %(message)s') # 정보 출력(시간, 메시지)
    ah.setFormatter(formatter) # 포매터를 일반 핸들러에도 붙이고 파일 핸들러에도 붙임.

    logger.addHandler(ah) # 로거에 만든 핸들러를 붙여줌.


    num_examples, train_batches, dev_batches, test_batches, embedding_matrix, vocab = load_data(config)
    print(embedding_matrix.shape)
    config.n_embed, config.d_embed = embedding_matrix.shape # config 추가(n_embed, d_embed)
    config.dim_hidden = config.dim_sem+config.dim_str # config 추가(hidden = semantic_dimension + structure_dimension)

    print(config.__flags)
    # example : {'rnn_cell': <absl.flags._flag.Flag object at 0x7f16402168d0>, 'data_file': <absl.flags._flag.Flag object at 0x7f1640216e80>, 'batch_size': <absl.flags._flag.Flag object at 0x7f16142b09b0>, 'epochs': <absl.flags._flag.Flag object at 0x7f16142b0b00>, 'dim_str': <absl.flags._flag.Flag object at 0x7f16142b0ba8>, 'dim_sem': <absl.flags._flag.Flag object at 0x7f16142b0c88>, 'dim_output': <absl.flags._flag.Flag object at 0x7f16142b0cc0>, 'keep_prob': <absl.flags._flag.Flag object at 0x7f16142b0dd8>, 'opt': <absl.flags._flag.Flag object at 0x7f16142b0f28>, 'lr': <absl.flags._flag.Flag object at 0x7f16142b0f60>, 'norm': <absl.flags._flag.Flag object at 0x7f16142bb048>, 'gpu': <absl.flags._flag.Flag object at 0x7f16142bb0b8>, 'sent_attention': <absl.flags._flag.Flag object at 0x7f16142bb0f0>, 'doc_attention': <absl.flags._flag.Flag object at 0x7f16142bb1d0>, 'large_data': <absl.flags._flag.BooleanFlag object at 0x7f16142b0b38>, 'log_period': <absl.flags._flag.Flag object at 0x7f16142bb2e8>}

    logger.critical(str(config.__flags))
    # 로그 출력 : logger.debug('debug'), logger.info('info'), logger.warn('warn'), logger.error('error'), logger.critical('critical')
    
    model = StructureModel(config) # config전달 받아서 model 객체를 생성한다.
    model.build()
    model.get_loss()
    # trainer = Trainer(config)

    num_batches_per_epoch = int(num_examples / config.batch_size) # num_examples : len(train), example: 32/16 = 2-> 한 epoch에 2배피씩 처리
    num_steps = config.epochs * num_batches_per_epoch # 30 * 2 = 60 -> 총 처리해야하는 스텝

    with tf.Session() as sess:
        gvi = tf.global_variables_initializer()
        sess.run(gvi)
        sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32))) # embedding_matrix : load_data에서 가지고 옴.
        loss = 0

        for ct, batch in tqdm.tqdm(train_batches, total=num_steps):
            feed_dict = model.get_feed_dict(batch)
            outputs,_,_loss = sess.run([model.final_output, model.opt, model.loss], feed_dict=feed_dict)
            loss+=_loss

            if(ct%config.log_period==0): # config,log_period = 5000
                acc_test = evaluate(sess, model, test_batches)
                acc_dev = evaluate(sess, model, dev_batches)
                print('Step: {} Loss: {}\n'.format(ct, loss))
                print('Test ACC: {}\n'.format(acc_test))
                print('Dev  ACC: {}\n'.format(acc_dev))
                logger.debug('Step: {} Loss: {}\n'.format(ct, loss))
                logger.debug('Test ACC: {}\n'.format(acc_test))
                logger.debug('Dev  ACC: {}\n'.format(acc_dev))
                logger.handlers[0].flush() # ?
                loss = 0 # 5000단위로 loss본다.

