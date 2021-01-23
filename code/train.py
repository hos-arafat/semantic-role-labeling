import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from stud.dataset_creation import PreProcessor, create_dataset
from stud.model_architectures import Baseline_SRL_Model, BERT_SRL_Model
from stud.training import Trainer
from stud.utilities import (collate_labelled_batch, compute_precision,
                            create_pretrained_embeddings,
                            load_pretrained_embeddings,
                            get_glove_embedding_dict)
from training import HParams, opts


# Set the random seed to be able to reproduce results when needed
random.seed(opts['random_seed'])
np.random.seed(opts['random_seed'])
torch.manual_seed(opts['random_seed'])

p = PreProcessor("train", opts)

# Load tokenizers for input, pos, and labels. If they don't exist, create them
if os.path.exists(p.word_to_int_path):
    (vocabulary, decode), (prd_vocabulary, prd_decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.load_all_tokenizers()
else:
    p.read_labelled_data("./data")
    (vocabulary, decode), (prd_vocabulary, prd_decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.create_all_tokenizers()

# Create train and dev datasets
train_dataset = create_dataset(dataset_type="train", soruce="./data", opts=opts)
train_dataloader = data.DataLoader(train_dataset, collate_fn=collate_labelled_batch, batch_size=opts["batch_size"], shuffle=True)


dev_dataset = create_dataset(dataset_type="dev", soruce="./data", opts=opts)
dev_dataloader = data.DataLoader(dev_dataset, collate_fn=collate_labelled_batch, batch_size=opts["batch_size"], shuffle=False)

print("\nVocab size",len(vocabulary))
print("POS Vocab size",len(pos_vocabulary))
print("Label Vocab size",len(label_vocabulary))

# Define hyperparameters
params = HParams(vocabulary, prd_vocabulary, pos_vocabulary, label_vocabulary, opts)

if opts['use_glove_embeddings'] == True:


    glove_embeddings_npy_path = "./model/glove_embeddings.npy"

    # If embeddings are saved as NPY files, load them. If not, create them
    if not os.path.exists(glove_embeddings_npy_path):
        glove_word2embed_dict = get_glove_embedding_dict()

        pretrained_embeddings = create_pretrained_embeddings("glove", glove_word2embed_dict, vocabulary, decode, opts["glove_embedding_dim"])
    else:
        pretrained_embeddings = load_pretrained_embeddings(glove_embeddings_npy_path).to(opts["device"])

    # Set model embeddings equal to the pretrained embeddings
    params.embeddings    = pretrained_embeddings

# Create the model according to the architecture defined in "opts" dictionary
if opts["use_bert"] == False:

    print("\n\nBiLSTM")
    srl_model = Baseline_SRL_Model(params).cuda()
    if opts['use_pos_embeddings'] == True:
        print("with POS embeddings")
    if opts['use_glove_embeddings'] == True:
        print("with GloVe embeddings")

elif opts["use_bert"] == True:
    print("\n\nBert")
    srl_model = BERT_SRL_Model(params).cuda()
    if opts['use_pos_embeddings'] == True:
        print("with POS embeddings")

# Create Trainer
trainer = Trainer(
    model         = srl_model,
    loss_function = nn.CrossEntropyLoss(ignore_index=label_vocabulary['<pad>']),
    optimizer     = optim.Adam(srl_model.parameters(), lr=opts['learning_rate']),
    label_vocab   = label_vocabulary,
    options          = opts
)

# Create folder to save the model in
if not os.path.exists(opts["save_model_path"]):
    os.makedirs(opts["save_model_path"])


print("Number of Epochs", opts["epochs"])

# Train model
trainer.train(train_dataloader, dev_dataloader)

# Create file to write all the model hyperparameters and training cofiguration in
hyperParams_file = open("{}/hyper-parameters.txt".format(opts["save_model_path"]), "a", encoding="utf8")
for (option, value) in opts.items():
    hyperParams_file.write("{}: {}\n".format(option, value))

parent = "./data"
p1 = PreProcessor("dev", opts)
_ = p1.read_labelled_data(parent) 

# Evaluate model by computing precision, recall & F-score on dev set
ident_precisions, cls_precisions = compute_precision(srl_model, dev_dataloader, label_vocabulary, int_to_label, opts, p1)

print("Identification\n\tPrecision: {}\n\tRecall: {}\n\tF-1 score: {}\n\n".format(ident_precisions["precision"], ident_precisions["recall"], ident_precisions["f1"]))
print("Classification\n\tPrecision: {}\n\tRecall: {}\n\tF-1 score: {}\n\n".format(cls_precisions["precision"], cls_precisions["recall"], cls_precisions["f1"]))

