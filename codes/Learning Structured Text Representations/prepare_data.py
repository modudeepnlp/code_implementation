from data_structure import Corpus
import argparse

import cPickle
# The cPickle module implements the same algorithm, in C instead of Python.

# data_structure 파일의 Corpus 클래스 객체를 이용해서 preprocess 진행
def main(train_path, dev_path, test_path):
    corpus = Corpus()
    corpus.load(train_path, 'train')
    corpus.load(dev_path, 'dev')
    corpus.load(test_path, 'test')
    corpus.preprocess()
    options =  dict(max_sents=60, max_tokens=100, skip_gram=False, emb_size=200)
    print('Start training word embeddings')
    corpus.w2v(options)

    instance, instance_dev, instance_test, embeddings, vocab = corpus.prepare(options)

    cPickle.dump((instance, instance_dev, instance_test, embeddings, vocab),open('../data/yelp-2013-all.pkl','w'))


# 먼저 argparse로 train_path, dev_path, test_path 불러들여온다. (데이터 가져오기)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('train_path', action="store")
parser.add_argument('dev_path', action="store")
parser.add_argument('test_path', action="store")
args = parser.parse_args()

# train_path = '../data/yelp-2013.train'
# dev_path = '../data/yelp-2013.dev'
# test_path = '../data/yelp-2013.test'

# 메인함수로 가져온 주소들을 넣어줌.
main(args.train_path, args.dev_path, args.test_path)