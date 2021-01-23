import os

import torch
from torch.utils import data

from stud.dataset_creation import PreProcessor, create_dataset
from stud.model_architectures import Baseline_SRL_Model, BERT_SRL_Model
from stud.training import HParams, opts
from stud.utilities import (collate_labelled_batch, compute_precision,
                            visualize_model_embeddings)

print("\nEvaluating {} model saved in {} at epoch {}\n".format(opts["architecture"], opts["save_model_path"], opts["epochs"] - 1))

device = opts["device"]

p = PreProcessor("train", opts)

# Load tokenizers for input, pos, and labels. If they don't exist, create them
if os.path.exists(p.word_to_int_path):
    (vocabulary, decode), (prd_vocabulary, prd_decode), (pos_vocabulary, pos_decode), (label_vocabulary, int_to_label) = p.load_all_tokenizers()
else:
    print("Can not evaluate model -that should have been trained on a specific vovabulary- if said vocabulary does not exist !")
    exit()

# Create dev and test datasets
dev_dataset = create_dataset(dataset_type="dev", soruce="./data", opts=opts)
dev_dataloader = data.DataLoader(dev_dataset, collate_fn=collate_labelled_batch, batch_size=256)


test_dataset = create_dataset(dataset_type="test", soruce="./data", opts=opts)
test_dataloader = data.DataLoader(test_dataset, collate_fn=collate_labelled_batch, batch_size=256)


print("Number of Epochs", opts["epochs"])

print("Loading trained model...")

# Define Hyperparameters
load_params = HParams(vocabulary, prd_vocabulary, pos_vocabulary, label_vocabulary, opts)



with torch.no_grad():
    # Instatiate appropriate model
    if opts["use_bert"] == False:
        print("\n\nBiLSTM")
        srl_model = Baseline_SRL_Model(load_params).cuda()
        if opts['use_pos_embeddings'] == True:
            print("with POS embeddings")
        if opts['use_glove_embeddings'] == True:
            print("with GloVe embeddings")

    elif opts["use_bert"] == True:
        print("\n\nBert")
        srl_model = BERT_SRL_Model(load_params).cuda()
        if opts['use_pos_embeddings'] == True:
            print("with POS embeddings")
    
    # Load model in eval mode
    state_dict = torch.load(os.path.join(opts["save_model_path"], 'state_{}.pth'.format(opts["epochs"]-1)), map_location=device)
    srl_model.load_state_dict(state_dict)    
    srl_model.to(device)
    srl_model.eval()

    print("Model loaded and set to EVAL mode") 

    # Plot word embeddings (and POS embeddings if model was trained with both)
    if opts["use_bert"] == False:
        word_embeds = srl_model.word_embedding.weight
        if opts["use_pos_embeddings"] == True:
            pos_embeds = srl_model.word_embedding.weight
            visualize_model_embeddings("pos", pos_embeds, pos_vocabulary, opts)

        visualize_model_embeddings("word", word_embeds, vocabulary, opts)

    
    parent = "./data"
    p1 = PreProcessor("test", opts)
    p1.read_labelled_data(parent)    

    # Evaluate model by computing precision, recall & F-score on dev or test set
    ident_precisions, cls_precisions = compute_precision(srl_model, test_dataloader, label_vocabulary, int_to_label, opts, p1)

    print("\n\nIdentification")
    print("\tPrecision: {}\n\tRcall: {}\n\tF-1 score: {}\n\n".format(ident_precisions["precision"], ident_precisions["recall"], ident_precisions["f1"]))


    print("\nClassification")
    print("\tPrecision: {}\n\tRcall: {}\n\tF-1 score: {}".format(cls_precisions["precision"], cls_precisions["recall"], cls_precisions["f1"]))
