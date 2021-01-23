import torch


def collate_test_batch(batch):
    """
    Function that pads a batch which DOES NOT contain ground truth labels
    
    Args:
        batch: list of tuples or dicts containing the inputs, POS tags
    Returns:
        padded_inputs: batch of torchTensors containing the input data
        padded_pos: batch of torchTensors containing the POS tags
        mask: batch of torchTensors reflecting seq length and padding with 1s and 0s
    """
    unpadded_inputs = []
    unpadded_preds = []
    unpadded_pos = []

    # Detect if batch has POS tags or not
    pos_present = False

    for tup in batch: 
        unpadded_inputs.append(tup["inputs"])
        unpadded_preds.append(tup["preds"])
        # If dict has POS Tag element
        if "pos" in tup:
            pos_present = True
            unpadded_pos.append(tup["pos"])

    # Pad inputs, predicates, and POS tags per batch with a value of 0
    padded_inputs = torch.nn.utils.rnn.pad_sequence(unpadded_inputs, batch_first=True, padding_value=0)
    padded_preds = torch.nn.utils.rnn.pad_sequence(unpadded_preds, batch_first=True, padding_value=0)
    mask = (padded_inputs != 0)
    if pos_present:
        padded_pos = torch.nn.utils.rnn.pad_sequence(unpadded_pos, batch_first=True, padding_value=0)
        return {"inputs":padded_inputs, "preds":padded_preds, "mask":mask, "pos":padded_pos}
    else:
        return {"inputs":padded_inputs, "preds":padded_preds, "mask":mask}





