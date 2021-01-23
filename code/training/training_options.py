# This file contains a dictionary that specifies the model's hyperparameters

opts = {}


opts['random_seed'] = 0

opts["task"] = "34" # "234" , "1234"
opts['use_binary_pred'] = False
opts['use_pred_embeddings'] = True

if opts["task"] == "34":
    if opts['use_binary_pred'] == True:
        opts['use_pred_embeddings'] = False
    if opts['use_pred_embeddings'] == True:
        opts['use_binary_pred'] = False

# Type of architecture to use
opts['architecture'] = "bilstm" # "lstm"

opts['use_bert'] = True
opts["bert_weights_path"] = "./model/transformers_bert"


opts['bidirectional'] = True
# Bidirectional LSTM or not
if opts['architecture'] == "bilstm" or opts['use_bert'] == True:
    opts['bidirectional'] = True
elif opts['architecture'] == "lstm" and opts['use_bert'] == False:
    opts['bidirectional'] = False


# Number of (Bi)LSTM layers
opts['lstm_layers'] = 2
if opts['use_bert'] == True:
    opts['lstm_layers'] = 1

# (Bi)LSTM hidden dimension
opts['hidden_dim'] = 256

opts['epochs'] = 3
opts['learning_rate'] = 0.001 
if opts['use_bert'] == True:
    opts['learning_rate'] = 5e-5

opts["batch_size"] = 256
if opts['use_bert'] == True:
    opts['batch_size'] = 32

opts['dropout'] = 0.4

# Wether or not to use pretrained GloVe embeddings
opts['use_glove_embeddings'] = False 
if opts['use_bert'] == True:
    opts['use_glove_embeddings'] = False

# Wether or not to use POS embeddings
opts['use_pos_embeddings'] = True

opts['glove_embedding_dim'] = 300
opts['pred_embedding_dim'] = 300
opts['pos_embedding_dim'] = 300

# Wether or not to use Gradient clipping
opts['grad_clipping'] = True

# Wether or not to use Early stopping
opts['early_stopping'] = True

opts['device'] = "cuda"

opts["save_model_path"] = "./model/BERT/Experiment_8"

