from torch.utils.data import Dataset

class SRLDataset(Dataset):
    """
    Class that creates a Semantic Role Labelling Dataset
    """ 
    def __init__(self, **kwargs):
        """    
        Args: 
            x: list of torchTensors representing the input
            pr: list of torchTensors representing the predicates
            p: list of torchTensors representing the POS tags
            y: list of torchTensors representing the labels
        
        Returns: 
            tuple or dict of torchTensors
        """
        self.encoded_data = kwargs["x"]
        self.encoded_preds = kwargs["pr"]
        self.encoded_pos = kwargs["p"]
        self.encoded_labels = kwargs["y"]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        t = []
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")
        
        # Data we get in "implementation.py" has no labels
        if self.encoded_labels is None: 
            # We experiment with feeding and NOT feeding POS Tags to model
            if self.encoded_pos is None: 
                return {"inputs": self.encoded_data[idx], "preds": self.encoded_preds[idx]}
            else:
                return {"inputs":self.encoded_data[idx], "preds": self.encoded_preds[idx], "pos":self.encoded_pos[idx]}
        else: 
            # Train-dev-test data does have labels
            if self.encoded_pos is None: 
            # We experiment with feeding and NOT feeding POS Tags to model
                return {"inputs":self.encoded_data[idx], "preds": self.encoded_preds[idx], "labels":self.encoded_labels[idx]}
            else:
                return {"inputs":self.encoded_data[idx], "preds": self.encoded_preds[idx], "labels":self.encoded_labels[idx], "pos":self.encoded_pos[idx]}
