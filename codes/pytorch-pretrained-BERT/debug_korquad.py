import os
import logging
import json
import collections

import examples.run_squad as run_squad
from pytorch_pretrained_bert.tokenization import BertTokenizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    input_file = "./samples/SQuAD/train-simple.json"
    # input_file = "./samples/KorQuAD/KorQuAD_simple_train.json"
    bert_model = "bert-base-multilingual-cased"
    do_lower_case = False
    max_seq_length = 128
    doc_stride = 128
    max_query_length = 64

    train_examples = run_squad.read_squad_examples(input_file=input_file, is_training=True, version_2_with_negative=False)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    run_squad.convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=True)

