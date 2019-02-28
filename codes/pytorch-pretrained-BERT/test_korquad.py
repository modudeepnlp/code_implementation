import os
import logging
import json
import collections

import torch
from torch.nn import BCEWithLogitsLoss

from khaiii import KhaiiiApi

from pytorch_pretrained_bert.modeling import BertConfig, BertPreTrainedModel, BertModel
import examples.run_squad as run_squad

api = KhaiiiApi()

def make_korquad_vocab(input_file, output_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def build_tokens(tokens, text):
        for word in api.analyze(text):
            for index in range(len(word.morphs)):
                if word.morphs[index].tag != "SN": ## 숫자는 제거
                    if index == 0:
                        token = word.morphs[index].lex
                    else:
                        token = "##" + word.morphs[index].lex
                    tokens[token] = token
    
    tokens = collections.OrderedDict()
    for i in range(10):
        i = str(i)
        tokens[i] = i

    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            # print(paragraph_text)
            build_tokens(tokens, paragraph_text)
            for qa in paragraph["qas"]:
                # print(qa["question"])
                build_tokens(tokens, qa["question"])
                for answer in qa["answers"]:
                    # print(answer["text"])
                    build_tokens(tokens, answer["text"])
    
    with open(output_file, "w", encoding='utf-8') as writer:
        for key, _ in tokens.items():
            writer.write(key)
            writer.write(os.linesep)
        print(len(tokens))


def make_simple_json(input_file, output_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    
    output_data = {}
    output_data["version"] = input_data["version"]
    datas = []
    for data in input_data["data"]:
        datas.append(data)
        break
    output_data["data"] = datas

    with open(output_file, "w", encoding='utf-8') as writer:
        json.dump(output_data, writer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    input_file = "./samples/KorQuAD/KorQuAD_v1.0_train.json"
    # make_korquad_vocab(input_file, "./samples/KorQuAD/vocab.txt")
    # make_simple_json(input_file, "./samples/KorQuAD/KorQuAD_simple_train.json")
    make_simple_json("./samples/SQuAD/train-v1.1.json", "./samples/SQuAD/train-simple.json")

    # examples = run_squad.read_squad_examples(input_file=input_file, is_training=True, version_2_with_negative=False)
    # for example in examples:
    #     print("==================================================")
    #     print(example.doc_tokens)
