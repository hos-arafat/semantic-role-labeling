import os
import pickle

import numpy as np
import torch
import torchtext


def save_to_pickle(path, item):
    """
    Function that saves an item to pickle file
    """
    with open(path, mode='wb') as f:
        pickle.dump(item, f)


def load_from_pickle(path):
    """
    Function that loads an item from pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def collate_labelled_batch(batch):
    """
    Function that pads a batch which contains ground truth labels

    Args:
        batch: list of tuples or dicts containing the inputs, POS tags and labels
    Returns:
        padded_inputs: batch of torchTensors containing the input data
        padded_preds: batch of torchTensors containing the predicate data
        padded_labels: batch of torchTensors containing the ground truth labels
        padded_pos: batch of torchTensors containing the POS tags
        mask: batch of torchTensors reflecting seq length and padding with 1s and 0s
    """

    unpadded_inputs = []
    unpadded_preds = []
    unpadded_pos = []
    unpadded_labels = []
    pred_idx = []

    # Detect if batch has POS tags or not
    pos_present = False

    for tup in batch:
        unpadded_inputs.append(tup["inputs"])
        unpadded_preds.append(tup["preds"])
        unpadded_labels.append(tup["labels"])
        # If tuple has POS Tag element
        if "pos" in tup:
            pos_present = True
            unpadded_pos.append(tup["pos"])


    # Pad inputs, POS tags, predicates, and labels per batch with a value of 0
    padded_inputs = torch.nn.utils.rnn.pad_sequence(unpadded_inputs, batch_first=True, padding_value=0)
    padded_preds = torch.nn.utils.rnn.pad_sequence(unpadded_preds, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(unpadded_labels, batch_first=True, padding_value=0)
    mask = (padded_inputs != 0)
    if pos_present:
        padded_pos = torch.nn.utils.rnn.pad_sequence(unpadded_pos, batch_first=True, padding_value=0)
        return {"inputs":padded_inputs, "preds":padded_preds, "labels":padded_labels, "mask":mask, "pos":padded_pos}
    else:
        return {"inputs":padded_inputs, "preds":padded_preds, "labels":padded_labels, "mask":mask}

def get_glove_embedding_dict():
    """
    Function that returns a dictionary containing GloVe vectors
    Args:

    Returns:
        index_to_vec: "int:[embedding]" dictionary mapping each token
        to its corresponding GloVe embedding vector
    """

    # create integer:embedding GloVe vector dictionary 
    index_to_vec = torchtext.vocab.GloVe(name="42B", dim=300)        

    return index_to_vec

def create_pretrained_embeddings(mode, token2vec_dict, vocabulary, decode, dim):
    """
    Function that creates pre-trained embeddings as a numpy array [vocab_size, embedding_dimension]
    
    Args:
        mode: name of pretrained embeddings to create
        token2vec_dict: "int:[embedding]" dictionary representing storing each token as an int
        and its corresponding embedding vector, as returned by the "parse_embeddings_text_file" function
        vocabulary: dictionary mapping token to ints
        decode: dictionary ints to tokens
        dim: embedding dimension
    Returns:
        pretrained_embeddings: a numpy array representing embeddings [vocab_size, embedding_dimension]
    """
    
    if mode == "glove":
        print("Loading GloVe embeddings from disk...")
        embeddings_npy_path = "./model/glove_embeddings.npy"

    # Create torchTensor with size [vocab size, embedding dimension]
    pretrained_embeddings = torch.randn(len(vocabulary), dim)
    initialised = 0

    # loop over int-to-token dictionary
    for (i, w) in (decode.items()):

        i = int(i)
        w = str(w)

        try:
            # Set the i-th element in embedding layer to be 
            # equal to the embedding of the i-th word in our vocab
            vec = token2vec_dict[w]
            initialised += 1
            pretrained_embeddings[i] = torch.FloatTensor(vec)
        except KeyError:
            continue            

    # Set 0th vector in embedding layer to be all 0s
    pretrained_embeddings[vocabulary["<pad>"]] = torch.zeros(dim)

    # Save embeddings as NPY for faster loading in the future
    # using the "load_pretrained_embeddings" function below
    np.save(embeddings_npy_path, pretrained_embeddings)
    
    print("Done! \nInitialised {} embeddings {}".format(mode, initialised))
    print("Random initialised embeddings {} ".format(len(vocabulary) - initialised))

    return pretrained_embeddings


def load_pretrained_embeddings(embeddings_npy_path):
    """
    Function that loads pre-trained embeddings as a numpy array [vocab_size, embedding_dimension]
    
    Args:
        embeddings_npy_path: path to NPY file to load

    Returns:
        pretrained_embeddings: a torch Tensor representing embeddings [vocab_size, embedding_dimension]
    """
    print("\nLoading Embeddings from NPY file")
    pretrained_embeddings = torch.LongTensor(np.load(embeddings_npy_path))
    print("Embeddings from NPY file shape", pretrained_embeddings.shape)

    return pretrained_embeddings
