import os
import logging

import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from pytorch_pretrained_bert.modeling import BertEmbeddings, BertSelfAttention, BertSelfOutput, BertAttention
from pytorch_pretrained_bert.modeling import BertIntermediate, BertOutput, BertLayer, BertEncoder, BertPooler
from pytorch_pretrained_bert.modeling import BertPredictionHeadTransform, BertLMPredictionHead, BertOnlyMLMHead
from pytorch_pretrained_bert.modeling import BertOnlyNSPHead, BertPreTrainingHeads
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertForMaskedLM
from pytorch_pretrained_bert.modeling import BertForNextSentencePrediction, BertForSequenceClassification
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertForTokenClassification, BertForQuestionAnswering

def test_BertEmbeddings():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    model = BertEmbeddings(config)
    print(model(input_ids, token_type_ids))


def test_BertSelfAttention():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertSelfAttention(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    input_mask = input_mask.view([-1, 1, 1, input_mask.size()[-1]]).float()
    print(model(embedding_output, input_mask))


def test_BertSelfOutput():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertSelfOutput(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output, embedding_output))


def test_BertAttention():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertAttention(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    input_mask = input_mask.view([-1, 1, 1, input_mask.size()[-1]]).float()
    print(model(embedding_output, input_mask))


def test_BertIntermediate():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertIntermediate(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output))


def test_BertOutput():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    intermediate = BertIntermediate(config)
    model = BertOutput(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    intermediate_output = intermediate(embedding_output)
    print(model(intermediate_output, embedding_output))


def test_BertLayer():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertLayer(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    input_mask = input_mask.view([-1, 1, 1, input_mask.size()[-1]]).float()
    print(model(embedding_output, input_mask))


def test_BertEncoder():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertEncoder(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    input_mask = input_mask.view([-1, 1, 1, input_mask.size()[-1]]).float()
    print(model(embedding_output, input_mask))


def test_BertPooler():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertPooler(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output))


def test_BertPredictionHeadTransform():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertPredictionHeadTransform(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output))


def test_BertLMPredictionHead():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertLMPredictionHead(config, embeddings.word_embeddings.weight)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output))


def test_BertOnlyMLMHead():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertOnlyMLMHead(config, embeddings.word_embeddings.weight)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output))


def test_BertOnlyNSPHead():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertOnlyNSPHead(config)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output))


def test_BertPreTrainingHeads():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    embeddings = BertEmbeddings(config)
    model = BertPreTrainingHeads(config, embeddings.word_embeddings.weight)

    embedding_output = embeddings(input_ids, token_type_ids)
    print(model(embedding_output, embedding_output))


def test_BertModel():
    model = BertModel.from_pretrained("bert-base-cased")
    # print(model)
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    print(model(input_ids, token_type_ids, input_mask))
    # print(len(all_encoder_layers))
    # for layer in all_encoder_layers:
    #     print(layer.size())
    #     # print(layer)
    # print("==================================")
    # print(pooled_output.size())
    # print(pooled_output)


def test_BertForPreTraining():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    print(model(input_ids, token_type_ids, input_mask))


def test_BertForMaskedLM():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    model = BertForMaskedLM(config)
    print(model(input_ids, token_type_ids, input_mask))


def test_BertForNextSentencePrediction():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    model = BertForNextSentencePrediction(config)
    print(model(input_ids, token_type_ids, input_mask))


def test_BertForSequenceClassification():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    print(model(input_ids, token_type_ids, input_mask))


def test_BertForMultipleChoice():
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    num_choices = 2
    model = BertForMultipleChoice(config, num_choices)
    print(model(input_ids, token_type_ids, input_mask))


def test_BertForTokenClassification():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForTokenClassification(config, num_labels)
    print(model(input_ids, token_type_ids, input_mask))


def test_BertForQuestionAnswering():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072)
    model = BertForQuestionAnswering(config)
    print(model(input_ids, token_type_ids, input_mask))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_BertEmbeddings()
    # test_BertSelfAttention()
    # test_BertSelfOutput()
    # test_BertAttention()
    # test_BertIntermediate()
    # test_BertOutput()
    # test_BertLayer()
    # test_BertEncoder()
    # test_BertPooler()
    # test_BertPredictionHeadTransform()
    # test_BertLMPredictionHead()
    # test_BertOnlyMLMHead()
    # test_BertOnlyNSPHead()
    # test_BertPreTrainingHeads()
    test_BertModel()
    # test_BertForPreTraining()
    # test_BertForMaskedLM()
    # test_BertForNextSentencePrediction()
    # test_BertForSequenceClassification()
    # test_BertForMultipleChoice()
    # test_BertForTokenClassification()
    # test_BertForQuestionAnswering()
