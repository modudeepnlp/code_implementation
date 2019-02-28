import os
import logging

import torch
from pytorch_pretrained_bert.optimization import BertAdam

def test_BertAdam():
    params = [
        torch.typename('lr'),
        # torch.ones((2,), dtype=torch.int8)
    ]
    optimizer = BertAdam(params)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_BertAdam()
