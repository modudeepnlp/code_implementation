import os
import logging

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining
from examples.run_lm_finetuning import BERTDataset


"""
* train_batch_size를 32에서 16으로 변경해야 함

python -m examples.run_lm_finetuning \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --do_train \
  --train_file ./samples/sample_text.txt \
  --output_dir ./samples/samples_out \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 128
"""
def run_lm_finetuning():
  pass


"""
python -m examples.extract_features \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --input_file ./samples/input.txt \
  --output_file ./samples/input_out.txt \
  --max_seq_length 128
"""
def extract_features():
  pass


"""
python -m examples.run_squad \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ./samples/SQuAD/train-v1.1.json \
  --predict_file ./samples/SQuAD/dev-v1.1.json \
  --train_batch_size 7 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./samples/output_squad/
"""
def run_squad():
  pass


"""
nohup python -m examples.run_squad \
  --bert_model bert-base-multilingual-cased \
  --do_train \
  --do_predict \
  --train_file ./samples/KorQuAD/KorQuAD_v1.0_train.json \
  --predict_file ./samples/KorQuAD/KorQuAD_v1.0_dev.json \
  --train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./samples/output_korquad/ &
"""
def run_korquad():
  pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    run_lm_finetuning()

 
