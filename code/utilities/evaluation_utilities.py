import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn

from utils import (evaluate_argument_classification,
                   evaluate_argument_identification)


def compute_precision(model:nn.Module, l_dataset, l_label_vocab, int_to_label, opts, pre):
    """
    Function that Computes precision, recall, F-score, and confusion matrix
    
    Args:
        model: pytroch model to evaluate
        l_dataset: dataloader of a labelled dataset
        l_label_vocab: dictionary mapping labels to ints
        int_to_label: dictionary mapping ints to labels
        opts: dictionary that specifies various training options and hyperparameters
        pre: PreProcessor class that has read labelled data from file
    Returns:
        dictionary contatining precision, recall, F-score, and confusion matrix
    """

    all_predictions = list()
    reconstructed_predictions = list()
    all_labels = list()
    
    read_labels = list()
    read_predictions = list()

    pos = None
    pred = None

    inp, prs = pre.list_l_sentences, pre.list_l_unpacked_predicates
    original_predicates = pre.list_l_original_predicates
    labels_as_dict = pre.labels_dict
    dev_d = pre.list_l_labels

    model.eval()

    duplicates = [(len(x) - x.count("_")) for x in original_predicates]

    pred_idx = [[i for i, v in enumerate(x) if v!="_"] for x in original_predicates]

    # Loop over dataset (dataloader)
    for batch in l_dataset:

        inputs = batch["inputs"].to(opts["device"])
        if opts["use_binary_pred"] == True:
            pred = batch["preds"].type(torch.FloatTensor).to(opts["device"])
        else:
            pred = batch["preds"].to(opts["device"])
        labels = batch["labels"].to(opts["device"])
        mask = batch["mask"].to(opts["device"], dtype=torch.uint8)
        
        # If model uses POS embeddings include batch of encoded POS tags
        if opts["use_pos_embeddings"] == True:
            pos = batch["pos"].to(opts["device"])

        # Get original length of every sentence in the batch
        # subtract 4 if we are using BERT (as we will remove the 4 special tokens in BERT)
        if opts["use_bert"] == False:
            original_len = np.asarray([len(x[x.nonzero(as_tuple=True)].tolist()) for x in inputs]) 
        elif opts["use_bert"] == True:
            unmodified_original_len = [len(x[x.nonzero(as_tuple=True)].tolist()) for x in inputs]
            original_len = np.asarray([b - 4 for b in unmodified_original_len])

        labels = labels.view(-1)
        valid_indices = labels != 0
        
        # Forward pass through network
        predictions = model(inputs, mask, pred, pos)
        predictions = torch.argmax(predictions, -1).view(-1)
        valid_predictions = predictions[valid_indices]

        # Map predictions from ints to words
        mapper = lambda x: int_to_label[x]
        mapped_predictions = list(map(mapper, valid_predictions.tolist()))
        all_predictions.extend(mapped_predictions)

        # Recontruct predictions from one big list that has predictions for entire batch
        # into a list of lists where every list has predictions for a sentence
        reconst = []                    
        var = 0
        for c_idx in range(len(original_len)):
            reconst.append(mapped_predictions[var:var+original_len[c_idx]])
            var += original_len[c_idx]

        reconstructed_predictions.extend(reconst)

        valid_labels = labels[valid_indices]
        all_labels.extend(valid_labels.tolist())
        
        read_labels.append(valid_labels.tolist())

    # Get index of a few labels and access the POS of those indeces
    select_labels_idx = [i for i, v in enumerate(all_labels) if v==4 or v==6 or v==10 or v==12 or v==16 or v==18]
    all_pos = [y for x in pre.list_l_pos for y in x]
    
    pos_that_are_agent = [all_pos[ag_ix] for ag_ix in select_labels_idx]
    labels_that_are_agent = [int_to_label[all_labels[ag_ix]] for ag_ix in select_labels_idx]
    
    # Create a datafram of POS Tags and the labels associated with them
    df = pd.DataFrame(list(zip(pos_that_are_agent, labels_that_are_agent)), columns=["POS Tag", "Label"])
    
    # Plot the POS Tags of the Labels we selected in "select_labels_idx"
    df=df.groupby(['POS Tag','Label']).size()
    df=df.unstack()
    df.plot(kind='bar')
    plt.xticks(rotation=0)
    plt.show()

    # Convert predictions from a list of lists into a dictionary:
    # sentence index: { roles: { index of predicate: [ labels of predicate ] } }
    all_predictions_dict = {}
    c = 0
    for idx, elem in enumerate(duplicates):
        all_predictions_dict.update({idx: {"roles": {}}})
        if elem == 0:
            c += 1
        else:    
            for h in range(elem):
                None if not pred_idx[idx] else all_predictions_dict[idx]["roles"].update({pred_idx[idx][h]:reconstructed_predictions[c]})
                c += 1

    # Compute precision, recall, F-score
    arg_ident = evaluate_argument_identification(labels_as_dict, all_predictions_dict)
    arg_class = evaluate_argument_classification(labels_as_dict, all_predictions_dict)

    # Write the ground truth label and prediction of every sentence to file
    # for debugging purposes
    eval_file = "{}/Prediction_vs_GroundTruth.txt".format(opts["save_model_path"])
    s_op_file = open(eval_file, "w", encoding="utf8")

    for true_label, prediction, sentence, pr  in zip(dev_d, reconstructed_predictions, inp, prs):
        
        s_op_file.write("sen:")
        for token in sentence:
            s_op_file.write(token + " ")
        s_op_file.write("\n")

        s_op_file.write("prd:")
        for th in pr:
            s_op_file.write(th + " ")
        s_op_file.write("\n")

        s_op_file.write("org:")
        for word in true_label:
            s_op_file.write(str(word) + " ")
        s_op_file.write("\n")

        s_op_file.write("pre:")
        for p in prediction:
            s_op_file.write(str(p) + " ")
        s_op_file.write("\n")
        s_op_file.write("\n")
        assert len(true_label) == len(prediction)
  
    # Write the Precision, Recall and F1-score the model acheived to file      
    s_op_file.write("\n\nIdentification")
    s_op_file.write("\n\tPrecision: {}\n\tRcall: {}\n\tF-1 score: {}\n\n".format(arg_ident["precision"], arg_ident["recall"], arg_ident["f1"]))


    s_op_file.write("\nClassification")
    s_op_file.write("\n\tPrecision: {}\n\tRcall: {}\n\tF-1 score: {}\n\n".format(arg_class["precision"], arg_class["recall"], arg_class["f1"]))

    return arg_ident, arg_class
