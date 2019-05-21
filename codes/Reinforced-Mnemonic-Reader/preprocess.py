
#%%
import json
from tqdm import tqdm
import spacy
import numpy as np
import re
from util.tokenizer import normalize_text, normal_query
from util.spacy_tokenizer import SpacyTokenizer
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from functools import partial
TOK = None
ANNTOTORS = {'lemma', 'pos', 'ner'}

def reform_text(text):
#     text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', token_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

TOK = SpacyTokenizer(annotators=ANNTOTORS)

def word_tokenize(text, norm=False):
    tokens = TOK.tokenize(reform_text(text)) # reform_text
    output = {
        'words': tokens.words(),
        'pos': tokens.pos(),
        'ner': tokens.entities(),
        'lemma': tokens.lemmas(),
    }
    if norm:
        output['words'] = [normalize_text(t) for t in output['words']]
    return output

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def process_file(filename, data_type, word_counter, char_counter, pos_counter, ner_counter):
    examples = []
    eval_examples = {}
    total = 0
    unans = 0
    ans = 0
    contexts = []
    questions = []
    # fetch contexts and questions
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in source["data"]:
            for para in article["paragraphs"]:
                contexts.append(para["context"].replace("''", '" ').replace("``", '" '))
                questions.append(para['qas'])
                
    print("Generating {} context...".format(data_type))            
    make_pool = partial(Pool, 12)
    workers = make_pool(initargs=())
    c_tokens = workers.map(word_tokenize, contexts)
    print("Generating {} context over".format(data_type)) 
    for i in tqdm(range(len(c_tokens))):
        ct = c_tokens[i]
        # get tokens, pos, ner, lemma
        context_tokens = ct['words']
        context_pos = ct['pos']
        context_ner = ct['ner']
        context_lemma = ct['lemma']
        spans = convert_idx(contexts[i], context_tokens)
        context_tokens = [normalize_text(t) for t in context_tokens]
        context_chars = [list(token) for token in context_tokens]
        for j in range(len(context_tokens)):
            word_counter[context_tokens[j]] += len(questions[i])
            pos_counter[context_pos[j]] += len(questions[i])
            ner_counter[context_ner[j]] += len(questions[i])
            for char in context_tokens[j]:
                char_counter[char] += len(questions[i])
        for qa in questions[i]:
            total += 1
            ques = qa["question"].replace("''", '" ').replace("``", '" ')
            qt = word_tokenize(ques, norm=True)
            ques_tokens = qt['words']
            # 预处理：替换question里context出现过的数字
            ques_tokens = normal_query(ques_tokens, context_tokens)
            ques_pos = qt['pos']
            ques_ner = qt['ner']
            ques_lemma = qt['lemma']
            ques_chars = [list(token) for token in ques_tokens]
            for j, token in enumerate(ques_tokens):
                word_counter[token] += 1
                pos_counter[ques_pos[j]] += 1
                ner_counter[ques_ner[j]] += 1
                for char in token:
                    char_counter[char] += 1
            y1s, y2s = [], []
            # 2.0 plausible answers
            y1sp, y2sp = [], []
            answer_texts = []
            
            # 2.0 Dataset
            if 'is_impossible' in qa and qa['is_impossible']==True:
                unans += 1
                for answer in qa["plausible_answers"]:
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    if len(answer_span)==0:
                        print(answer,answer_text)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1sp.append(y1)
                    y2sp.append(y2)
                y1s.append(-1)
                y2s.append(-1)
            else:
                ans += 1
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    if len(answer_span)==0:
                        print(answer,answer_text)
#                     else:
#                         print(answer_text, '###', np.array(context_tokens)[answer_span])
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                    y1sp.append(y1)
                    y2sp.append(y2)
            example = {"context_tokens": context_tokens, "context_chars": context_chars, 
                       'context_lemma':context_lemma, 'context_pos':context_pos, 'context_ner':context_ner,
                       "ques_tokens": ques_tokens, "ques_chars": ques_chars, 
                       'ques_lemma':ques_lemma, 'ques_pos':ques_pos, 'ques_ner':ques_ner,
                       "y1s": y1s, "y2s": y2s, 
                       'y1sp':y1sp, 'y2sp':y2sp, 
                       "id": total}
            examples.append(example)
            eval_examples[str(total)] = {"question":ques,
                                         "context": contexts[i], 
                                         "spans": spans, 
                                         "answers": answer_texts, 
                                         "uuid": qa["id"]}
    print("{} questions in total".format(len(examples)))
    print('answerable:',ans,'unanswerable:',unans)
    
    return examples, eval_examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if data_type=='char':
        embedding_dict_fix={}
        embedding_dict_trainable={}
        assert size is not None
        assert vec_size is not None
        assert emb_file is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                char = "".join(array[0:-vec_size])
                char = normalize_text(char)
                vector = list(map(float, array[-vec_size:]))
                if char in counter and counter[char] > limit:
                    embedding_dict_fix[char] = vector
        print("{} / {} char tokens have corresponding {} embedding vector".format(
                len(embedding_dict_fix), len(filtered_elements), data_type))
        for token in filtered_elements:
            if token not in embedding_dict_fix:
                embedding_dict_trainable[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        
        # trainable emb mat
        NULL = "--NULL--"
        OOV = "--OOV--"
        token2idx_dict = {token: idx for idx,
                          token in enumerate(embedding_dict_trainable.keys(), 2)}
        token2idx_dict[NULL] = 0
        token2idx_dict[OOV] = 1
        embedding_dict_trainable[NULL] = [0. for _ in range(vec_size)]
        embedding_dict_trainable[OOV] = [0. for _ in range(vec_size)] # np.random.random((vec_size))/2-0.25
        idx2emb_dict = {idx: embedding_dict_trainable[token]
                        for token, idx in token2idx_dict.items()}
        emb_mat_trainable = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
        
        # fix emb mat
        for idx, token in enumerate(embedding_dict_fix.keys(), len(token2idx_dict)):
            token2idx_dict[token] = idx
        for token, idx in token2idx_dict.items():
            if token not in embedding_dict_trainable:
                idx2emb_dict[idx] = embedding_dict_fix[token]
        emb_mat_fix = [idx2emb_dict[idx] for idx in range(len(emb_mat_trainable), len(idx2emb_dict))]
        print('idx2emb_dict:',len(idx2emb_dict))
        print('token2idx_dict:',len(token2idx_dict))
        print('emb_mat_trainable:',len(emb_mat_trainable))
        print('emb_mat_fix:',len(emb_mat_fix))
        return (emb_mat_trainable, emb_mat_fix), token2idx_dict, idx2emb_dict
    else:
        embedding_dict={}
        if emb_file is not None:
            assert size is not None
            assert vec_size is not None
            with open(emb_file, "r", encoding="utf-8") as fh:
                for line in tqdm(fh, total=size):
                    array = line.split()
                    word = "".join(array[0:-vec_size])
                    word = normalize_text(word)
                    vector = list(map(float, array[-vec_size:]))
                    if word in counter and counter[word] > limit:
                        embedding_dict[word] = vector
            print("{} / {} word tokens have corresponding {} embedding vector".format(
                len(embedding_dict), len(filtered_elements), data_type))
        else:
            assert vec_size is not None
            for token in filtered_elements:
                embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
            print("{} char tokens have corresponding embedding vector".format(
                len(filtered_elements)))

        NULL = "--NULL--"
        OOV = "--OOV--"
        token2idx_dict = {token: idx for idx,
                          token in enumerate(embedding_dict.keys(), 2)}
        token2idx_dict[NULL] = 0
        token2idx_dict[OOV] = 1
        embedding_dict[NULL] = [0. for _ in range(vec_size)]
        embedding_dict[OOV] = [0. for _ in range(vec_size)] # np.random.random((vec_size))/2-0.25
        idx2emb_dict = {idx: embedding_dict[token]
                        for token, idx in token2idx_dict.items()}
        emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
        return emb_mat, token2idx_dict, idx2emb_dict


#%%
from collections import Counter
import numpy as np
word_counter, char_counter = Counter(), Counter()

# # 2.0 Dataset
# test_examples, test_eval = process_file('original_data/dev-v2.0.json', "test", word_counter, char_counter)
# train_examples, train_eval = process_file('original_data/train-v2.0.json', "train", word_counter, char_counter)

# 1.0 Dataset
train_examples, train_eval = process_file('../../fwei/data/squad/train-v1.1.json', "train", word_counter, char_counter)
# dev_examples, dev_eval = process_file('../../fwei/data/squad/dev-v1.2.json', "dev", word_counter, char_counter)
test_examples, test_eval = process_file('../../fwei/data/squad/dev-v1.1.json', "test", word_counter, char_counter)


#%%
# save train_eval and dev_eval
# # 2.0 Dataset
# with open('dataset/train_eval.json', "w") as fh:
#     json.dump(train_eval, fh)
# with open('dataset/test_eval.json','w') as fh:
#     json.dump(test_eval,fh)
    
# 1.0 Dataset
with open('dataset1.0/train_eval.json', "w") as fh:
    json.dump(train_eval, fh)
# with open('dataset1.0/dev_eval.json','w') as fh:
#     json.dump(dev_eval,fh)
with open('dataset1.0/test_eval.json','w') as fh:
    json.dump(test_eval,fh)


#%%
from collections import Counter
import numpy as np
word_counter, char_counter = Counter(), Counter()
pos_counter, ner_counter = Counter(), Counter()
# 2.0 Dataset
test_examples, test_eval = process_file('original_data/dev-v1.1.json', "test", 
                                        word_counter, char_counter, pos_counter, ner_counter)
train_examples, train_eval = process_file('original_data/train-v1.1.json', "train",
                                          word_counter, char_counter, pos_counter, ner_counter)


#%%
# save word_tokens
all_tokens=set(list(word_counter.keys()))
print('token num:', len(all_tokens))
assert '<S>' not in all_tokens and '</S>' not in all_tokens
all_tokens.add('<S>')
all_tokens.add('</S>')
print('token + <S> + </S> num:', len(all_tokens))
for a in all_tokens:
    if a=='':
        print('nono')
vocab_file = '../RMR_tf/dataset2/vocab.txt'
with open(vocab_file, 'w',encoding='utf8') as fout:
    fout.write('\n'.join(all_tokens))


#%%
# save train_eval and dev_eval
# 2.0 Dataset
with open('../RMR_tf/dataset2/train_eval.json', "w") as fh:
    json.dump(train_eval, fh)
with open('../RMR_tf/dataset2/test_eval.json','w') as fh:
    json.dump(test_eval,fh)


#%%
# w2v 90977 91586
(char_embmat_trainable, char_embmat_fix), char2idx_dict, _ = get_embedding(
    char_counter, "char", emb_file='original_data/glove.840B.300d-char.txt', size=95, vec_size=300)
word_emb_mat, word2idx_dict, _ = get_embedding(
    word_counter, "word", emb_file='original_data/glove.840B.300d.txt', size=int(2.2e6), vec_size=300)


#%%
# get pos and ner embedding
def get_tag_emb(counter):
    emb_dict={}
    max_len=len(counter)
    for i,c in enumerate(counter):
        emb_vec = np.zeros(max_len)
        emb_vec[i]=1
        emb_dict[c]=emb_vec
    print('emb_dict size:',len(emb_dict))
    return emb_dict

pos_emb=get_tag_emb(pos_counter)
ner_emb=get_tag_emb(ner_counter)
print('all pos:',pos_counter.keys())
print('all ner:',ner_counter.keys())


#%%
import pickle
import h5py
def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, id2word_dict,                    pos_emb, ner_emb, is_test=False):

    para_limit = config['test_para_limit'] if is_test else config['para_limit']
    ques_limit = config['test_ques_limit'] if is_test else config['ques_limit']
    ans_limit = 100 if is_test else config['ans_limit']
    char_limit = config['char_limit']
    
    def match_func(question, context, question_lemma, context_lemma):
        counter = Counter(w.lower() for w in context)
        total = sum(counter.values())
        freq = [counter[w.lower()] / total for w in context]
        question_word = {w for w in question}
        question_lower = {w.lower() for w in question}
        question_lemma = {w if w != '-PRON-' else w.lower() for w in question_lemma}
        match_origin = [1 if w in question_word else 0 for w in context]
        match_lower = [1 if w.lower() in question_lower else 0 for w in context]
        match_lemma = [1 if (w if w != '-PRON-' else w.lower()) in question_lemma else 0 for w in context_lemma]
        features = np.asarray([freq, match_origin, match_lower, match_lemma], dtype=np.float32).T
        return features

    def filter_func(example, is_test=False):
        if len(example['y2s'])==0 or len(example['y1s'])==0:
            print(example)
        return len(example["context_tokens"]) > para_limit or                len(example["ques_tokens"]) > ques_limit or                (example["y2s"][0] - example["y1s"][0]) > ans_limit
    
    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    qids=[]
    context_strings_all=[]
    ques_strings_all=[]
    unans=0
    with h5py.File(out_file+data_type+'_data.h5','w') as h5f:
        for example in tqdm(examples):
            total_ += 1

            if filter_func(example, is_test):
                continue

            total += 1
            qids.append(str(example['id']))
            c_len = len(example['context_tokens'])
            q_len = len(example['ques_tokens'])
            context_idxs = []
            context_char_idxs = np.zeros([c_len, char_limit], dtype=np.int32)
            ques_idxs = []
            ques_char_idxs = np.zeros([q_len, char_limit], dtype=np.int32)
            context_strings = []
            ques_strings = []
            context_pos = []
            context_ner = []
            ques_pos = []
            ques_ner = []

            if config['data_ver']==2:
                y1 = np.zeros([c_len+1], dtype=np.float32)
                y2 = np.zeros([c_len+1], dtype=np.float32)
                y1p = np.zeros([c_len], dtype=np.float32)
                y2p = np.zeros([c_len], dtype=np.float32)
            else:
                y1 = np.zeros([c_len], dtype=np.float32)
                y2 = np.zeros([c_len], dtype=np.float32)
                y1p = None
                y2p = None

            for i, token in enumerate(example["context_tokens"]):
                context_idxs.append(_get_word(token))
                context_strings.append(token)
            context_idxs=np.array(context_idxs)

            for i, token in enumerate(example["ques_tokens"]):
                ques_idxs.append(_get_word(token)) 
                ques_strings.append(token)
            ques_idxs=np.array(ques_idxs)

            for i, token in enumerate(example["context_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idxs[i, j] = _get_char(char)

            for i, token in enumerate(example["ques_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[i, j] = _get_char(char)

            for i, token in enumerate(example["context_pos"]):
                context_pos.append(pos_emb[token])
            context_pos=np.array(context_pos)
            
            for i, token in enumerate(example["context_ner"]):
                context_ner.append(ner_emb[token])
            context_ner=np.array(context_ner)
            
            context_match = match_func(example["ques_tokens"], example["context_tokens"], 
                                       example["ques_lemma"], example["context_lemma"])
            context_feat = np.concatenate([context_pos, context_ner, context_match], axis=-1) # [c_len, 50+18+4]

            for i, token in enumerate(example["ques_pos"]):
                ques_pos.append(pos_emb[token])
            ques_pos=np.array(ques_pos)
            
            for i, token in enumerate(example["ques_ner"]):
                ques_ner.append(ner_emb[token])
            ques_ner=np.array(ques_ner)
            
            ques_match = match_func(example["context_tokens"], example["ques_tokens"], 
                                          example["context_lemma"], example["ques_lemma"])
            ques_feat = np.concatenate([ques_pos, ques_ner, ques_match], axis=-1) # [q_len, 50+18+4]


            start, end = example["y1s"][-1], example["y2s"][-1]
            if config['data_ver']==2: 
                if len(example["y1sp"])!=0:
                    startp, endp = example["y1sp"][-1], example["y2sp"][-1]
                if start!=-1 and end!=-1:
                    y1[start+1], y2[end+1] = 1.0, 1.0
                    y1p[start], y2p[end] = 1.0, 1.0
                else:
                    y1[0], y2[0] = 1.0, 1.0
                    if len(example["y1sp"])!=0:
                        y1p[startp], y2p[endp] = 1.0, 1.0
                    unans+=1
            else:
                y1[start], y2[end] = 1.0, 1.0
            
            data_simple = h5f.create_group(str(example['id']))
            data_simple.create_dataset('context_ids', data = context_idxs)
            data_simple.create_dataset('ques_ids', data = ques_idxs)
            data_simple.create_dataset('context_char_ids', data = context_char_idxs)
            data_simple.create_dataset('ques_char_ids', data = ques_char_idxs)
            data_simple.create_dataset('y1', data = y1)
            data_simple.create_dataset('y2', data = y2)
            if config['data_ver']==2:
                data_simple.create_dataset('y1p', data = y1p)
                data_simple.create_dataset('y2p', data = y2p)
            data_simple.create_dataset('context_feat', data = context_feat)
            data_simple.create_dataset('ques_feat', data = ques_feat)
            context_strings_all.append(context_strings)
            ques_strings_all.append(ques_strings)
            
    with open(out_file+data_type+'_contw_strings.pkl','wb') as f:
        pickle.dump(context_strings_all, f)
    with open(out_file+data_type+'_quesw_strings.pkl','wb') as f:
        pickle.dump(ques_strings_all, f)
        
    np.save(out_file+data_type+'_qid.npy',qids)
    
    print("Built {} / {} instances of features in total".format(total, total_))
    print('unanswerable:',unans)

config={
    'test_para_limit':1000,
    'test_ques_limit':50,
    'para_limit':400,
    'ques_limit':50,
    'ans_limit':30,
    'char_limit':16,
    'data_ver':1,
    'typo_correct':True
}

# 2.0 Dataset
build_features(config, train_examples, 'train', '../RMR_tf/dataset2/', word2idx_dict, char2idx_dict, id2word_dict, 
               pos_emb, ner_emb, is_test=False)
build_features(config, test_examples, 'dev', '../RMR_tf/dataset2/', word2idx_dict, char2idx_dict, id2word_dict, 
               pos_emb, ner_emb, is_test=False)
# build_features(config, test_examples, 'test', 'dataset_pre3/', word2idx_dict, char2idx_dict, id2word_dict, 
#                pos_emb, ner_emb, is_test=True)


#%%


data_type='dev'
cont_string=np.load(os.path.join('dataset',data_type+'_contw_strings.npy'))

import spacy
nlp = spacy.load("en")
print([i.tag_ for i in nlp('cannot')])
# words=cont_string[0,:]
# print(x)
# tags_=[]
# for w in words:
#     wtag=[j.tag_ for j in nlp(str(w))]
#     tags_.append(wtag)
# print(words)
# print(tags_)
nlp = spacy.load("en")
from tqdm import tqdm

def gettag(cont_string):
    contexts=[]
    wrong_num=0
    for i in tqdm(range(cont_string.shape[0])):#range(cont_string.shape[0])
        sentences=[]
        words=[]
        for j in range(cont_string.shape[1]):
            if cont_string[i,j]=='':
                break

            # 规则矫正：
            # 1.如果只有一个'，去除
            if str(cont_string[i,j]).count('\'')==1 and len(cont_string[i,j])>1:
                cont_string[i,j]=cont_string[i,j].replace('\'','')
                
            # 2.如果如果是cannot，改为not
            if str(cont_string[i,j]).lower()=='cannot':
                cont_string[i,j]='not'
                
            # 3.im，改为I
            if str(cont_string[i,j]).lower()=='im':
                cont_string[i,j]='i'
                
            # # 其余问题过滤（暂时）128K
            # if str(cont_string[i,j])=='128K':
            #     cont_string[i,j]='128'
                
            words.append(cont_string[i,j])
            if words[-1]=='.' or words[-1]=='!' or words[-1]=='?':
                sentence=' '.join(words)
                tags=[n.tag_ for n in nlp(sentence)]
                if len(tags)!=len(words):
                    tags_=[]
                    for w in words:
                        wtag=[j.tag_ for j in nlp(str(w))]
                        if len(wtag)>1:
                            wtag=wtag[0]
                        tags_.extend(wtag)
                    tags=tags_
                    wrong_num+=1
                    assert len(tags)==len(words)
                sentences.append(list(zip(tags,words)))
                words=[]
        if len(words)>0:
            sentence=' '.join(words)
            tags=[n.tag_ for n in nlp(sentence)]
            if len(tags)!=len(words):
                tags_=[]
                for w in words:
                    wtag=[j.tag_ for j in nlp(str(w))]
                    if len(wtag)>1:
                        wtag=wtag[0]
                    tags_.extend(wtag)
                tags=tags_
                wrong_num+=1
                assert len(tags)==len(words)
            sentences.append(list(zip(tags,words)))
            words=[]
        contexts.append(sentences)
    print(wrong_num)
    
    return contexts
# contexts=gettag(cont_string)

split_num=8
temp_len=cont_string.shape[0]//split_num
params=[]
for i in range(split_num):
    if i != split_num-1:
        params.append(cont_string[i*temp_len:(i+1)*temp_len,::])
    else:
        params.append(cont_string[i*temp_len:,::])
    
from multiprocessing import Pool
pool=Pool()
result=[]
for i in params:
    result.append(pool.apply_async(gettag, kwds={'cont_string':i}))
pool.close()
pool.join()
contexts=[]
[contexts.extend(i.get()) for i in result]
import torch
import parse_nk
torch.cuda.set_device(3)
def torch_load(load_path):
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)
info = torch_load('parsing/models/en_elmo_dev.95.21.pt')
assert 'hparams' in info['spec'], "Older savefiles not supported"
info['spec']['hparams']['sentence_max_len']=400
print(info['spec']['hparams'])
parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])


#%%
import pickle
import numpy as np
with open('parsing/data/dev_tags.pkl','rb') as f:
    tags=pickle.load(f)
    tags=np.array(tags)
tags_temp=tags[64:96]


#%%
import numpy as np

def generate_parse_feat(tags_temp):
    batch_size = len(tags_temp)
    
    # stastic the word num in each sample
    sen_len=[sum([len(tt) for tt in t]) for t in tags_temp]
    max_len=max(sen_len)
    
    # combine the sentences to a batch
    tags_temp_new=[]
    for i in range(len(tags_temp)):
        combined_context=[]
        [combined_context.extend(t) for t in tags_temp[i]]
        tags_temp_new.append(combined_context)
    print(len(tags_temp_new[11]))
    # inference the parsing feature
    feat,idxs = parser.parse_batch(tags_temp_new)
    
    # remove the elmo useless token from feat
    inds=[]
    for j in range(len(idxs.batch_idxs_np)):
        if j==0 or j==len(idxs.batch_idxs_np)-1 or         idxs.batch_idxs_np[j-1]!=idxs.batch_idxs_np[j] or         idxs.batch_idxs_np[j+1]!=idxs.batch_idxs_np[j]:
            continue
        else:
            inds.append(j)
    feat=feat[inds,:]
    
    # convert feat to (batch_size, max_len, 1024)
    assert sum(sen_len)==feat.shape[0]
    feats=np.zeros((batch_size, max_len, 1024))
    cusum=0
    for i,s in enumerate(sen_len):
        feats[i,0:s,:]=feat[cusum:cusum+s,:]
        cusum+=s
    assert cusum==feat.shape[0]
    
    return feats


#%%



#%%
nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]
words = word_tokenize("im a footman.")

print(words)


