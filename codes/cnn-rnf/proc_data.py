#-*- coding: utf-8 -*-
# Copyright 2018 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cPickle
import argparse, logging

logger = logging.getLogger("cnn_rnf.proc_data")

def build_data(fnames):
    """
    Load and process data.
    """
    ## fnames = [train, dev, test]
    revs = []
    vocab = set()
    corpora = []
    ## fnames의 corpus를 corpora에 읽어들임
    for i in xrange(len(fnames)):
        corpora.append(get_corpus(fnames[i]))
    max_l = 0
    for i, corpus in enumerate(corpora):
        for [label, words] in corpus: ## (label, [문자열 array])
            for word in words:
                vocab.add(word)
            datum  = {'y':label,
                      'words': words,
                      'num_words': len(words),
                      'split': i} ## index of corpus
            max_l = max(max_l, datum['num_words']) ## 최대 tokern 개수
            revs.append(datum)
    logger.info("vocab size: %d, max sentence length: %d" %(len(vocab), max_l))
    return revs, vocab, max_l
   
## 파일을 읽어서 (label, [문자열 array]) 향태로 데이터 생성
def get_corpus(fname):
    corpus = []
    with open(fname, 'rb') as f:
        for line in f:
            line = line.strip()
            line = line.replace("-lrb-", "(")
            line = line.replace("-rrb-", ")")
            parts = line.split()
            corpus.append((int(parts[0]), parts[1:]))
    return corpus


class WordVecs(object):
    """
    Manage precompute embeddings
    """
    ## fname: pretrained word vector (Glove)
    ## vocab: set of vocab
    def __init__(self, fname, vocab, random=True):
        word_vecs, self.k = self.load_txt_vec(fname, vocab)
        self.random = random
        self.add_unknown_words(word_vecs, vocab)
        self.W, self.word_idx_map = self.get_W(word_vecs)

    ## word vector index map 및 word vector matrix 생성
    def get_W(self, word_vecs):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, self.k))            
        W[0] = np.zeros(self.k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map
   
    def load_txt_vec(self, fname, vocab): ## file glove vocab file
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            parts = header.strip().split()
            ## 이 로직은 1번만 실행 되며 아래 'for line in f'에서 나머지는 실행 됨
            if len(parts) == 2:
                vocab_size, word_dim = map(int, parts) ## 이 경우는 발생하지 않음
            else:
                word = parts[0]
                if word in vocab: ## vocab 에 있는 경우만 vector 저장
                   word_vecs[word] = np.asarray(map(float, parts[1:])) ## key: word, value: [vector array] 형태로 저장
                word_dim = len(parts) - 1 ## word를 제외한 dimension
            for line in f: ## 첫번째 라인 이후 전체 라인에 애해 실행
                parts = line.strip().split()
                word = parts[0]
                if word in vocab: ## vocab 에 있는 경우만 vector 저장
                   word_vecs[word] = np.asarray(map(float, parts[1:])) ## key: word, value: [vector array] 형태로 저장
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs, word_dim 

    ## word_vecs에 oov 단어 추가
    def add_unknown_words(self, word_vecs, vocab):
        """
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs: ## 벡터에 존재 하지 않으면
                if self.random:
                    word_vecs[word] = np.random.uniform(-0.25, 0.25, self.k)  ## 렌덤인 경우는 -0.25 ~ 0.25 사이의 값으로 초기화
                else:
                    word_vecs[word] = np.zeros(self.k) ## 렌덤이 아닌 경우는 0으로 초기화


## args 분석 및 읽어들임
def parse_args():
    parser = argparse.ArgumentParser(description='VAEs')
    parser.add_argument('--train-path', type=str, default='data/sst_text_convnet/stsa.binary.phrases.train', help="path for training data")
    parser.add_argument('--dev-path', type=str, default='data/sst_text_convnet/stsa.binary.dev', help="path for development data")
    parser.add_argument('--test-path', type=str, default='data/sst_text_convnet/stsa.binary.test', help="path for test data")
    parser.add_argument('--emb-path', type=str, default='data/glove.840B.300d.txt', help="path for pretrained glove embbeddings")
    parser.add_argument('output', type=str, help="path for output pickle file")
    args = parser.parse_args()
    return args
 
 ## main 함수
def main():
    args = parse_args()
    revs, vocab, max_l = build_data([args.train_path, args.dev_path, args.test_path])
    logger.info("loading and processing pretrained word vectors")
    wordvecs = WordVecs(args.emb_path, vocab)
    cPickle.dump([revs, wordvecs, max_l], open(args.output, 'wb'))
    logger.info("dataset created!")

## debug용 parse_args
def debug_parse_args():
    parser = argparse.ArgumentParser(description='VAEs')
    parser.add_argument('--train-path', type=str, default='data/sst_text_convnet/stsa.binary.phrases.train', help="path for training data")
    parser.add_argument('--dev-path', type=str, default='data/sst_text_convnet/stsa.binary.dev', help="path for development data")
    parser.add_argument('--test-path', type=str, default='data/sst_text_convnet/stsa.binary.test', help="path for test data")
    parser.add_argument('--emb-path', type=str, default='data/glove.840B.300d.txt', help="path for pretrained glove embbeddings")
    # parser.add_argument('output', type=str, help="path for output pickle file")
    args = parser.parse_args()
    return args
 
 ## debug용 main
def debug_main():
    args = debug_parse_args()
    output = "data/stsa.binary.pkl"
    revs, vocab, max_l = build_data([args.train_path, args.dev_path, args.test_path])
    logger.info("loading and processing pretrained word vectors")
    wordvecs = WordVecs(args.emb_path, vocab)
    cPickle.dump([revs, wordvecs, max_l], open(output, 'wb')) ## output에 Pickle 형태로 저장
    logger.info("dataset created!")


## main
if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')
    # main()
    debug_main()
    logger.info("end logging")
