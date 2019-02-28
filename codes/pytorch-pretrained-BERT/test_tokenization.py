import os
import logging
import unicodedata
from pytorch_pretrained_bert import tokenization
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer, WordpieceTokenizer

model_dir = "/Users/cchyun/Dev/Git/Github/Dnn/pytorch-pretrained-BERT/bert-base-cased"


def test_BertTokenizer():
    model = BertTokenizer(os.path.join(model_dir, "bert-base-cased-vocab.txt"))


def test_BasicTokenizer():
    model = BasicTokenizer()
    ## 한글지원 안함
    text = "안녕hello월드ahahah國民mrhs"
    print(model._tokenize_chinese_chars(text))
    print(unicodedata.normalize("NFD", text))


def test_WordpieceTokenizer():
    model = WordpieceTokenizer(tokenization.load_vocab(os.path.join(model_dir, "bert-base-cased-vocab.txt")))
    print(model.tokenize("decomposition deoomposition"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_BertTokenizer()
    # test_BasicTokenizer()
    test_WordpieceTokenizer()
