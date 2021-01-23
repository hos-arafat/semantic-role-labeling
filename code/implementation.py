import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils import data

from model import Model

from stud.dataset_creation import PreProcessor, create_dataset
from stud.model_architectures import Baseline_SRL_Model, BERT_SRL_Model
from stud.training import HParams, opts
from stud.utilities import collate_test_batch

np.random.seed(opts["random_seed"])
torch.manual_seed(opts["random_seed"])

def build_model_34(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(device, opts) # is this okay ?
    # return StudentModel(device) # or should it be like this ??

def build_model_234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError

def build_model_1234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, return_predicates=False):
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence['pos_tags']:
            prob = self.baselines['predicate_identification'][pos]['positive'] / self.baselines['predicate_identification'][pos]['total']
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)
        
        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(zip(sentence['lemmas'], predicate_identification)):
            if not is_predicate or lemma not in self.baselines['predicate_disambiguation']:
                predicate_disambiguation.append('_')
            else:
                predicate_disambiguation.append(self.baselines['predicate_disambiguation'][lemma])
                predicate_indices.append(idx)
        
        argument_identification = []
        for dependency_relation in sentence['dependency_relations']:
            prob = self.baselines['argument_identification'][dependency_relation]['positive'] / self.baselines['argument_identification'][dependency_relation]['total']
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(sentence['dependency_relations'], argument_identification):
            if not is_argument:
                argument_classification.append('_')
            else:
                argument_classification.append(self.baselines['argument_classification'][dependency_relation])
        
        if self.return_predicates:
            return {
                'predicates': predicate_disambiguation,
                'roles': {i: argument_classification for i in predicate_indices},
            }
        else:
            return {'roles': {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path='data/baselines.json'):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device, options):
        
        self.device = device
        self.options = options

        self.parser = PreProcessor("submit", opts)

        
        self.use_pos = False 
        if self.options["use_pos_embeddings"] == True:
            self.use_pos = True
        
        self.bert_based = False 
        if self.options["use_bert"] == True:
            self.bert_based = True

        (vocabulary, decode), (prd_vocabulary, prd_decode), (pos_vocabulary, pos_decode), (label_vocabulary, self.int_to_label) = self.parser.load_all_tokenizers()
        load_params = HParams(vocabulary, prd_vocabulary, pos_vocabulary, label_vocabulary, self.options)
        
        if self.bert_based == False:
            print("\n\nBiLSTM")
            model = Baseline_SRL_Model(load_params)
        elif self.bert_based == True:
            print("\n\nBert")
            model = BERT_SRL_Model(load_params)

        self.srl_model = self.load_model(model)

    def load_model(self, model):
        save_path = self.options["save_model_path"]
        epoch = self.options["epochs"]

        state_dict = torch.load(os.path.join(save_path, 'state_{}.pth'.format(epoch-1)), map_location=self.device)
        model.load_state_dict(state_dict)
        
        return model


    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "pos_tags":
                            ["IN", "DT", "NN", ",", "NNP", "NNP", "CC", "DT", "NNS", "VBP", "IN", "DT", "JJ", "NNS", "IN", "VBG", "DT", "NN", "NN", "VBP", "RB", "VBN", "VBN", "."],
                        "dependency_heads":
                            ["10", "3", "1", "10", "6", "10", "6", "9", "7", "0", "10", "14", "14", "20", "14", "15", "19", "19", "16", "11", "20", "20", "22", "10"],
                        "dependency_relations":
                            ["ADV", "NMOD", "PMOD", "P", "TITLE", "SBJ", "COORD", "NMOD", "CONJ", "ROOT", "OBJ", "NMOD", "NMOD", "SBJ", "NMOD", "PMOD", "NMOD", "NMOD", "OBJ", "SUB", "TMP", "VC", "VC", "P"],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence. 
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence. 
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence. 
                    }
        """
        # Get length of sentence
        original_len = len(sentence["words"])

        reconstructed_predictions = list()
        all_predictions_dict = {}
        all_predictions_dict.update({"roles": {}})

        # Create dataset that has n copies of sentence
        # where n is the number of predicates if sentence has n > 1 predicates
        # or n = 1 if the sentence has 0 predicates
        submit_dataset, original_predicates = create_dataset(dataset_type="submit", soruce=sentence, opts=self.options)
        # Create dataloader
        submit_dataloader = data.DataLoader(submit_dataset, collate_fn=collate_test_batch, batch_size=1)

        # Get number of predicates sentence has
        num_of_predicates = [(len(x) - x.count("_")) for x in original_predicates]
        
        # If the sentence has 0 predicates, keep its roles dictionary empty
        if num_of_predicates[0] == 0:
            return all_predictions_dict
        
        elif num_of_predicates[0] > 0:
        
            pred_idx = [[i for i, v in enumerate(x) if v!="_"] for x in original_predicates]

            self.srl_model.to(self.device)
            self.srl_model.eval()

            with torch.no_grad():

                for batch_idx, batch in enumerate(submit_dataloader):

                    inputs = batch["inputs"].to(self.device)
                    if self.options["use_binary_pred"] == True:
                        pred = batch["preds"].type(torch.FloatTensor).to(self.device)
                    else:
                        pred = batch["preds"].to(self.device)

                    mask = None
                    pos = None
                    if self.bert_based == True:
                        mask = batch["mask"].to(self.device, dtype=torch.uint8)                
                    if self.use_pos == True:
                        pos = batch["pos"].to(self.device)
                    
                    # Get predictions from the model
                    predictions = self.srl_model(inputs, mask, pred, pos)
                    predictions = torch.argmax(predictions, -1).view(-1)
                    
                    # Map predictions from ints to words
                    mapper = lambda x: self.int_to_label[x]
                    mapped_predictions = list(map(mapper, predictions.tolist()))

                    # Add predictions to list
                    reconstructed_predictions.append(mapped_predictions)

                    # Update the predictions dictionary in the following way
                    # roles: { index of predicate: [ labels of predicate ] }
                    all_predictions_dict["roles"].update({pred_idx[0][batch_idx]:reconstructed_predictions[batch_idx]})
                    
            return all_predictions_dict