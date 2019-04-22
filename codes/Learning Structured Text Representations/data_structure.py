import gensim
import numpy as np
import re
import random
import math
import unicodedata
import itertools
from utils import grouper

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', unicode(s,'utf-8'))
                  if unicodedata.category(c) != 'Mn')


# class : RawData, DataSet, Instance, Corpus (Corpus부터 살펴보기)

class RawData:
    def __init__(self):
        self.userStr = ''
        self.productStr = ''
        self.reviewText = ''
        self.goldRating = -1
        self.predictedRating = -1
        self.userStr = ''


class DataSet:
    def __init__(self, data):
        self.data = data
        self.num_examples = len(self.data)

    def sort(self):
        random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x._max_sent_len)
        self.data = sorted(self.data, key=lambda x: x._doc_len)

    def get_by_idxs(self, idxs):
        return [self.data[idx] for idx in idxs]

    def get_batches(self, batch_size, num_epochs=None, rand = True):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        idxs = list(range(self.num_examples))
        _grouped = lambda: list(grouper(idxs, batch_size))

        if(rand):
            grouped = lambda: random.sample(_grouped(), num_batches_per_epoch)
        else:
            grouped = _grouped
        num_steps = num_epochs*num_batches_per_epoch
        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for i in range(num_steps):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs) # 데이터에서 해당 인덱스 범위에 있는 데이터 리스트 불러오기 
            yield i,batch_data
        # for ~ yield : 

class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

    def _doc_len(self, idx):
        k = len(self.token_idxs)
        return k

    def _max_sent_len(self, idxs):
        k = max([len(sent) for sent in self.token_idxs])
        return k

# Corpus : attribute(doclst--dictionary), method(load, preprocess, build_vocab, w2v, prepare, prepare_notest, prepare_for_training, prepare_for_test)
# load, preprocess, w2v method부터 살펴보기

# data example
# 0<split1>This place was very good just 5 years ago !<split2>Now , worst in the valley .<split2>Chips are bad - they came to our table and looked double fried ... just full of grease .<split2>We asked the server for fresh ones .<split2>He said that they only fry chips in the morning .<split2>He said he should warm these up ... he took the chips that we were picking at and put them into the fryer again ... gross and yuk !<split2>.<split2>.<split2>Salsa is old and was setting on the table when we got there ... as it was on every other table with no one there .<split2>Ours had started fermenting .<split2>Over all just a real lack of caring here anymore .<split2>Very sad ... oh very sad !!<split2>Wo n't every be back !<split2>
# 4<split1>One of the best hyatt in phoenix .<split2>The nice thing about this place is that the pool is all in one area , so you do n't have to run around to look for someone .<split2>Yea , its not as great as other resort -lrb- hilton has a nicer pool area -rrb- but if you can get a suite , its not a bad spot for a weekend vacation .<split2>The regency club is nice .<split2>Have a great selection of snack -lrb- breakfast , and deserts -rrb-<split2>
# 4<split1>Every time they are spot on ... thank you : - -rrb-<split2>
class Corpus:
    def __init__(self):
        self.doclst = {}

    def load(self, in_path, name):
        self.doclst[name] = []
        for line in open(in_path): # 라인 별로 불러오기
            items = line.split('<split1>')
            doc = RawData()
            doc.goldRating = int(items[0]) # 각 라인 별 split1 앞 부분
            doc.reviewText = items[1] # 각 라인 별 split1 뒷 부분
            self.doclst[name].append(doc) # dictionary value list에 객체 append 
            # 예시 : {'train': [<__main__.doc object at 0x7fcdfb5c6320>, <__main__.doc object at 0x7fcdfb5c6320>]}

    def preprocess(self):
        random.shuffle(self.doclst['train']) # 리스트 원소 랜덤 섞기
        for dataset in self.doclst:
            for doc in self.doclst[dataset]: # train 데이터들의 순서를 섞은 후에, 각 문서 별로 진행
                doc.sent_lst = doc.reviewText.split('<split2>') # 각 문서별로 split2 기준으로 문장 분리
                doc.sent_lst = [re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sent) for sent in doc.sent_lst] # 분리한 문장마다 정규표현식 적용
                # re.sub(pattern, repl, string) : string에서 pattern과 매치하는 텍스트를 repl로 치환한다
                doc.sent_token_lst = [sent.split() for sent in doc.sent_lst] # doc.sent_lst의 정규표현식으로 처리된 문장들을 하나씩 불러내서 token 생성
                doc.sent_token_lst = [sent_tokens for sent_tokens in doc.sent_token_lst if(len(sent_tokens)!=0)] # sent_tokens별로 길이가 0이 되지않도록 설정
            self.doclst[dataset] = [doc for doc in self.doclst[dataset] if len(doc.sent_token_lst)!=0]

    def build_vocab(self):
        self.vocab = {}
        for doc in self.doclst:
            for sent in doc.sent_token_lst:
                for token in sent:
                    if(token not in self.vocab):
                        self.vocab[token] = {'idx':len(self.vocab), 'count':1}
                    else:
                        self.vocab[token]['count'] += 1

    def w2v(self, options): # {'max_sents': 60, 'max_tokens': 100, 'skip_gram': False, 'emb_size': 200}
        
        sentences = []
        # train, dev에 해당되는 문서들의 sent_token_lst 불러서 sentences에 추가(list extend - list append와는 다름)
        # list append & extend : [1, 2, 3, [4, 5]] & [1, 2, 3, 4, 5]
        for doc in self.doclst['train']:
            sentences.extend(doc.sent_token_lst)
        if('dev' in self.doclst):
            for doc in self.doclst['dev']:
                sentences.extend(doc.sent_token_lst) 
        print(sentences[0])

        if(options['skip_gram']):
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4, sg=1)
        else: # CBOW model 
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4)
        self.w2v_model.scan_vocab(sentences)  # initial survey
        rtn = self.w2v_model.scale_vocab(dry_run = True)  # trim by min_count & precalculate downsampling
        print(rtn)

        self.w2v_model.finalize_vocab()  # build tables & arrays
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        self.vocab = self.w2v_model.wv.vocab
        print('Vocab size: {}'.format(len(self.vocab)))

        # model.save('../data/w2v.data')

    def prepare(self, options): # {'max_sents': 60, 'max_tokens': 100, 'skip_gram': False, 'emb_size': 200}
        instances, instances_dev, instances_test = [],[],[]
        instances, embeddings, vocab = self.prepare_for_training(options)
        if ('dev' in self.doclst):
            instances_dev = self.prepare_for_test(options, 'dev')
        instances_test = self.prepare_for_test( options, 'test')
        return instances, instances_dev, instances_test, embeddings, vocab

    def prepare_notest(self, options):
        instances, instances_dev, instances_test = [],[],[]
        instances_, embeddings, vocab = self.prepare_for_training(options)
        print(len(instances))
        for bucket in instances_:
            num_test = len(bucket) / 10
            instances_test.append(bucket[:num_test])
            instances.append(bucket[num_test:])

        return instances, instances_dev, instances_test, embeddings, vocab


    def prepare_for_training(self, options):
        instancelst = []
        embeddings = np.zeros([len(self.vocab)+1,options['emb_size']]) # shape: (단어 수, 임베딩 사이즈)
        for word in self.vocab: # 각 단어 별로 w2v 모델에서 훈련된 임베딩 벡터 가져오기
            embeddings[self.vocab[word].index] = self.w2v_model[word]
        self.vocab['UNK'] = gensim.models.word2vec.Vocab(count=0, index=len(self.vocab))
        n_filtered = 0
        for i_doc, doc in enumerate(self.doclst['train']):
            # 각 문서별로 본다
            instance = Instance()
            instance.idx = i_doc
            n_sents = len(doc.sent_token_lst)
            max_n_tokens = max([len(sent) for sent in doc.sent_token_lst])
            # 옵션에서 지정해놓은 sents, tokens갯수보다 크면 n_filtered 수 추가하기
            if(n_sents>options['max_sents']):
                n_filtered += 1
                continue
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue
            # token index vocab에 있으면 가져오고, 없으면 UNK 인덱스를 가져와서 붙임, 만든 문장 별 token 묶음 -> sent_token_idx append    
            sent_token_idx = []
            for i in range(len(doc.sent_token_lst)):
                token_idxs = []
                for token in doc.sent_token_lst[i]:
                    if(token in self.vocab):
                        token_idxs.append(self.vocab[token].index)
                    else:
                        token_idxs.append(self.vocab['UNK'].index)
                sent_token_idx.append(token_idxs)
            instance.token_idxs = sent_token_idx
            instance.goldLabel = doc.goldRating
            instancelst.append(instance)
        print('n_filtered in train: {}'.format(n_filtered))
        return instancelst, embeddings, self.vocab

    def prepare_for_test(self, options, name):
    # prepare_for_training과 비교 : embedding, vocab 부분이 제외됨.
        instancelst = []
        n_filtered = 0
        for i_doc, doc in enumerate(self.doclst[name]):
            instance = Instance()
            instance.idx = i_doc
            n_sents = len(doc.sent_token_lst)
            max_n_tokens = max([len(sent) for sent in doc.sent_token_lst])
            if(n_sents>options['max_sents']):
                n_filtered += 1
                continue
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue
            sent_token_idx = []
            for i in range(len(doc.sent_token_lst)):
                token_idxs = []
                for token in doc.sent_token_lst[i]:
                    if(token in self.vocab):
                        token_idxs.append(self.vocab[token].index)
                    else:
                        token_idxs.append(self.vocab['UNK'].index)
                sent_token_idx.append(token_idxs)
            instance.token_idxs = sent_token_idx
            instance.goldLabel = doc.goldRating
            instancelst.append(instance)
        print('n_filtered in {}: {}'.format(name, n_filtered))
        return instancelst