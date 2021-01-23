import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertForTokenClassification


class Baseline_SRL_Model(nn.Module):
    """Class the implementing our Baseline model"""
    def __init__(self, hparams):
        super(Baseline_SRL_Model, self).__init__()
        """
        Args:
            hparams: Class containing model hyperparameters
        Returns:
            o: output of the model
        """

        self.hparams = hparams

        # input to lstm is either (lemma + predicate) embeddings or (lemma + pos + predicate) embeddings
        lstm_input_dim = hparams.embedding_dim       


        # Lemma Embedding layer: a matrix [lemmas vocab size, embedding_dim]
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("\nModel Initializing lemma embeddings from pretrained GloVe")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        # Predicate Embedding layer: a matrix [Predicates vocab size, embedding_dim]
        if self.hparams.use_pred_embeddings == True:
            self.pred_embedding = nn.Embedding(self.hparams.pred_vocab_size, self.hparams.pred_embedding_dim)
            
            lstm_input_dim += hparams.pred_embedding_dim      
        # If we are using the Binary predicate indicator, increase BiLSTM input by 1
        elif self.hparams.use_binary_predicate == True:
            lstm_input_dim += 1      



        # POS Embedding layer: a matrix [POS tags vocab size, embedding_dim]
        if self.hparams.use_pos_embeddings == True:
            self.pos_embedding = nn.Embedding(self.hparams.pos_vocab_size, self.hparams.pos_embedding_dim)
            
            lstm_input_dim += hparams.pos_embedding_dim    

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with lemma embeddings) from left to right and outputs 
        # a new representation of each lemma 
        self.lstm = nn.LSTM(lstm_input_dim, hparams.hidden_dim, 
                            batch_first=True,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        # Hidden layer: transforms the input value/scalar into a hidden vector representation.
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        # Dropout and final fully connected layer
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

        print("LSTM input :", lstm_input_dim)

    
    def forward(self, x, _, pred, p):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        
        # Either concatenate predicate embedding to lemma embeddings
        # or concat the binary embedding directly to lemma embeddings 
        if self.hparams.use_pred_embeddings == True:    
            pred_indicator = self.pred_embedding(pred)
            pred_indicator = self.dropout(pred_indicator)
        else:
            pred_indicator = torch.unsqueeze(pred, -1)

        lemma_pred_embeddings = torch.cat((embeddings, pred_indicator), dim=-1)

        # If we are using POS embeddings 
        # concat POS embeddings to lemma+pred embedding and pass to LSTM
        if p is not None:
            pos_embeddings = self.pos_embedding(p)
            pos_embeddings = self.dropout(pos_embeddings)
            concat_embeddings = torch.cat((lemma_pred_embeddings, pos_embeddings), dim=-1)
            o, (h, c) = self.lstm(concat_embeddings)
        elif p is None:
            o, (h, c) = self.lstm(lemma_pred_embeddings)

        o = self.dropout(o)
        output = self.classifier(o)
        return output

        
class BERT_SRL_Model(nn.Module):
  def __init__(self, hparams):
    super(BERT_SRL_Model, self).__init__()

    self.hparams = hparams
    
    path_to_bert = "./model/transformers_bert"
    self.bert = BertModel.from_pretrained(path_to_bert)

    lstm_input_dim = self.bert.config.hidden_size

    # Predicate Embedding layer: a matrix [Predicates vocab size, embedding_dim]
    if self.hparams.use_pred_embeddings == True:
        self.pred_embedding = nn.Embedding(self.hparams.pred_vocab_size, self.hparams.pred_embedding_dim)
        
        lstm_input_dim += hparams.pred_embedding_dim    

    # POS Embedding layer: a matrix [POS vocab size, embedding_dim]
    if self.hparams.use_pos_embeddings == True:
        self.pos_embedding = nn.Embedding(self.hparams.pos_vocab_size, self.hparams.pos_embedding_dim)
        
        lstm_input_dim += hparams.pos_embedding_dim  


    self.lstm = nn.LSTM(lstm_input_dim, self.hparams.hidden_dim, 
                        batch_first=True,
                        bidirectional=self.hparams.bidirectional,
                        num_layers=self.hparams.num_layers, 
                        dropout = self.hparams.dropout if self.hparams.num_layers > 1 else 0)
    
    lstm_output_dim = self.hparams.hidden_dim if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2

    self.dropout = nn.Dropout(self.hparams.dropout)
    self.classifier = nn.Linear(lstm_output_dim, self.hparams.num_classes)

  
  def forward(self, input_ids, attention_mask, pred, pos):
    
    bert_hidden_layer, _ = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    # Get all non-zero elements in the input
    non_zeros = (input_ids != 0).sum(dim=1).tolist()

    # Drop the BERT embedding for '[CLS]' and '[SEP] predicate [SEP]'
    reconst = []
    for idx in range(len(non_zeros)):
        reconst.append(bert_hidden_layer[idx, 1:non_zeros[idx]-3, :])

    # Pad the tensor again after dropping '[CLS]' and '[SEP] predicate [SEP]'
    padded_again = torch.nn.utils.rnn.pad_sequence(reconst, batch_first=True, padding_value=0)

    pred_embeddings = self.pred_embedding(pred)
    
    # If we are NOT using POS embeddings 
    # concat BERT output to pred embedding and pass to LSTM
    if pos is None:
        concat_embeddings = torch.cat((padded_again, pred_embeddings), dim=-1)

    # If we are using POS embeddings 
    # concat POS embeddings to BERT embedding and pass to LSTM
    elif pos is not None:
        pos_embeddings = self.pos_embedding(pos)
        pos_embeddings = self.dropout(pos_embeddings)

        pos_pred = torch.cat((pos_embeddings, pred_embeddings), dim=-1)

        concat_embeddings = torch.cat((padded_again, pos_pred), dim=-1)

    o, (h, c) = self.lstm(concat_embeddings)
    output = self.classifier(o)
    return output
