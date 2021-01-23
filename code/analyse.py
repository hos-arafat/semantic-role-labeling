import json
import os
import sys

import torch
import torchtext
import torch.nn as nn

from stud.training import HParams, opts
from stud.utilities import visualize_model_embeddings
from transformers import BertModel, BertTokenizer

# Load BERT Model and Tokenizer
path_to_bert = "./model/transformers_bert"

tokenizer = BertTokenizer.from_pretrained(path_to_bert)
model = BertModel.from_pretrained(path_to_bert)

# 4 copies of sentence with 3 predicates to analyze difference in contextualized embeddings
# when we add NO or a different embedding after the [SEP] token
list_l_tokens = [ ['[CLS]', 'furukawa', 'electric', 'co.', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', 'u.s.', 'and', 'japan', '.', '[SEP]'],
['[CLS]', 'furukawa', 'electric', 'co.', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', 'u.s.', 'and', 'japan', '.', '[SEP]', 'say', '[SEP]'],
['[CLS]', 'furukawa', 'electric', 'co.', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', 'u.s.', 'and', 'japan', '.', '[SEP]', 'plan', '[SEP]'],
['[CLS]', 'furukawa', 'electric', 'co.', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', 'u.s.', 'and', 'japan', '.', '[SEP]', 'increase', '[SEP]'] ]

# Encode sentences using BERT Tokenizer
data = []
for sentence in list_l_tokens:
    encoding = tokenizer.encode_plus(
    sentence,
    add_special_tokens=False,
    return_token_type_ids=False,
    return_attention_mask=False,
    return_tensors='pt',  # Return PyTorch tensors
    )
    data.append(torch.squeeze(encoding['input_ids']))

# Pad encoded sentences
padded_inputs = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
mask = padded_inputs != 0

# Pass encoded sentences to BERT
bert_hidden_layer, _ = model(input_ids=padded_inputs, attention_mask=mask)

# 
non_zeros = (padded_inputs != 0).sum(dim=1).tolist()

# Remove the '[CLS]' and '[SEP]' tokens from first sentence 
# and the '[CLS]' '[SEP] predicate [SEP]' tokens from the rest 
reconst = []
for idx in range(len(non_zeros)):
    if idx == 0:
        reconst.append(bert_hidden_layer[idx, 1:non_zeros[idx]-1, :])
    else:
        reconst.append(bert_hidden_layer[idx, 1:non_zeros[idx]-3, :])

# Pad after removing the special tokens
padded_again = torch.nn.utils.rnn.pad_sequence(reconst, batch_first=True, padding_value=0)

# Reshape into a 2D Tensor
reshaped = torch.reshape(padded_again, (-1, 768))
# Plot
visualize_model_embeddings("bert", reshaped, tokenizer.get_vocab(), opts)

