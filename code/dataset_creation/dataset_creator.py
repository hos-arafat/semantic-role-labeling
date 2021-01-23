from stud.dataset_creation import PreProcessor, SRLDataset
from stud.training import opts


def create_dataset(dataset_type, soruce, opts):
    """
    Function that creates train-dev-test or submit SRL Dataset

    Args: 
        dataset_type: train or dev or test or submit if we are running in "implementation.py"
        source: soruce to read data from:
                                         The path to the train-dev-test data
                                         sentence we receive in "implementation.py"
        opts: dictionary outlining various options including if we want POS tags or not
    
    Returns: 
        dataset: SRLDataset instance
        p.list_l_original_predicates: Gold predicates
    """ 

    p = PreProcessor(dataset_type, opts)

    # If we are NOT running "implementation.py", we read the data from file
    if dataset_type == "train" or dataset_type == "dev" or dataset_type == "test":
        path_to_data = soruce
        p.read_labelled_data(path_to_data) 
    # Otherwise, we read the sentence that "implementation.py" gave us
    elif dataset_type == "submit":
        submission_sentence = soruce
        p.read_test_data(submission_sentence)

    # Encode all the data to a list of torchTensors
    encoded_tokens, encoded_pred, encoded_tokens_pos, encoded_labels = p.encode_all_data()
    # Create SRL dataset
    dataset = SRLDataset(x=encoded_tokens, pr=encoded_pred, p=encoded_tokens_pos, y=encoded_labels)
    print("{} dataset size is {}".format(dataset_type, len(dataset)))

    if dataset_type == "train" or dataset_type == "dev" or dataset_type == "test":
        return dataset
    elif dataset_type == "submit":
        return dataset, p.list_l_original_predicates