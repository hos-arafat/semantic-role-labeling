import json
import os
import pickle
from collections import OrderedDict

import torch
from transformers import AutoTokenizer


class PreProcessor():
    """Utility class to preprocess the train, dev, test files
       as well as the secret dataset we will get in 'implementation.py'."""
    def __init__(self, mode, opts):
        super(PreProcessor).__init__()
        """
        Args:
            mode: 'train', 'test', 'dev' OR 'submit' if we are running "implementation.py"
            opts: dictionary outlining various options
        """ 

        self.mode = mode
        # Specify which SRL tasks to do "34", "234", or "1234"
        # Currently, only task "34" is supported
        self.task = opts["task"]

        # Whether or not to use POS Tags
        self.include_pos = opts["use_pos_embeddings"]
        # Whether or not to use the binary predicate indicator
        self.binary_pred = opts["use_binary_pred"]

        # Whether or not to use BERT
        self.bert = opts["use_bert"]
        self.bert_path = opts["bert_weights_path"]


        if self.mode == "train":
            self.json_file = "train.json"

        if self.mode == "dev":
            self.json_file = "dev.json"
        
        if self.mode == "test":
            self.json_file = "test.json"
        
        # Path to the various input, POS, predicate, and label tokenizers
        self.word_to_int_path = os.path.join("./model/", "word_to_int_dictionary.pickle")
        self.int_to_word_path = os.path.join("./model/", "int_to_word_dictionary.pickle")

        self.pos_to_int_path = os.path.join("./model/", "pos_to_int_dictionary.pickle")
        self.int_to_pos_path = os.path.join("./model/", "int_to_pos_dictionary.pickle")

        self.predicate_to_int_path = os.path.join("./model/", "predicate_to_int_dictionary.pickle")
        self.int_to_predicate_path = os.path.join("./model/", "int_to_predicate_dictionary.pickle")

        self.label_to_int_path = os.path.join("./model/", "labels_to_int_dictionary.pickle")
        self.int_to_label_path = os.path.join("./model/", "int_to_labels_dictionary.pickle")

        self.training_tokens = list ()
        self.training_pos = list ()
        self.training_labels = list ()
        if self.task == "34":
            self.training_predicates = list ()

    def read_labelled_data(self, path_to_data):
        """
        Function that reads data file and creates list of lists 
        representing the input sentences, pos, predicates, and labels in the file

        Args:
            path_to_data: path to .json file
        
        Returns: None             
        """ 
        
        print("{:} Dataset will be built from the following files {:}...".format(self.mode, self.json_file))
    
        with open(os.path.join(path_to_data, self.json_file), "r", encoding="utf8") as f:
            dataset = json.load(f)

        self.labels_dict = dict()

        self.list_l_sentences = list()
        self.list_l_pos = list()
        self.list_l_labels =  list()
        if self.task == "34":
            self.list_l_unpacked_predicates = list()
            self.list_l_original_predicates = list()

        # Collect predicates and roles into a dictionary. 
        # Needed for calculating Precision, Recall, and F1 score
        for sentence_id, sentence in dataset.items():
            sentence_id = int(sentence_id)
            self.labels_dict[sentence_id] = {
                'predicates': sentence['predicates'],
                'roles': {int(p): r for p, r in sentence['roles'].items()}
            }

            # create lemma, POS, predicate, and label vocab
            self.training_tokens.extend(sentence["lemmas"])
            self.training_pos.extend(sentence["pos_tags"])
            self.training_labels.extend([t for p, r in sentence['roles'].items() for t in r])
            if self.task == "34":
                self.training_predicates.extend(sentence["predicates"])
                self.list_l_original_predicates.append(sentence['predicates'])

            # If sentence has 0 predicates
            if len(sentence["roles"].items()) == 0:

                empty_predicate = ['<pad>'] * len(sentence["lemmas"])

                if self.bert == False:
                    self.list_l_sentences.append(sentence["lemmas"])

                # If we are using BERT
                # Append Special [CLS] and [SEP] 0 [SEP] tokens
                elif self.bert == True:
                    copy_sent = list(sentence["lemmas"])
                    copy_sent.insert(0, "[CLS]")
                    copy_sent.append("[SEP]")
                    copy_sent.append("<pad>")
                    copy_sent.append("[SEP]")
                    self.list_l_sentences.append(copy_sent)                        

                self.list_l_unpacked_predicates.append(empty_predicate)
                self.list_l_pos.append(sentence['pos_tags'])  
                self.list_l_labels.append(sentence['predicates'])
            else:
                # If sentence has 1 or more predicates
                for k, v in sentence["roles"].items():

                    predicate = ["<pad>"] * len(sentence["lemmas"])
                    
                    if self.bert == False:
                        self.list_l_sentences.append(sentence["lemmas"])

                    # If we are using BERT
                    # Append Special [CLS] and [SEP] k-th predicate [SEP] tokens
                    elif self.bert == True:
                        copy_sent = list(sentence["lemmas"])
                        copy_sent.insert(0, "[CLS]")
                        copy_sent.append("[SEP]")
                        copy_sent.append(sentence["lemmas"][int(k)])
                        copy_sent.append("[SEP]")
                        self.list_l_sentences.append(copy_sent)
                    
                    # If we are using binary predicate indicator
                    # insert 1 at index of each predicate, 
                    # else, for predicate embeding, insert the predicate itself                    
                    if self.binary_pred == True:
                        predicate[int(k)] = "<unk>"
                    elif self.binary_pred == False:
                        predicate[int(k)] = sentence["predicates"][int(k)] 

                    self.list_l_pos.append(sentence['pos_tags'])   
                    self.list_l_unpacked_predicates.append(predicate)
                    self.list_l_labels.append(v)

        return

    def read_test_data(self, sentence): 
        """
        Function that reads sentence from 'implemntation.py' and creates list of lists 
        representing the input sentences, pos, predicates, and labels in the file

        Args:
            sentence: a dictionary containing the following:
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
            }            
        
        Returns: None             
        """      

        self.list_l_sentences = list()
        self.list_l_pos = list()

        if self.task == "34":
            self.list_l_unpacked_predicates = list()
            self.list_l_original_predicates = list()

        self.list_l_original_predicates.append(sentence['predicates'])

        # Get the number and indeces of all predicates in the sentence
        num_pred = len([x for x in sentence["predicates"] if x != "_"])
        pred_idx = [i for i, v in enumerate(sentence["predicates"]) if v!= "_"]

        # If sentence has 0 predicates
        if num_pred == 0:

            empty_predicate = ['<pad>'] * len(sentence["lemmas"])
            
            if self.bert == False:
                self.list_l_sentences.append(sentence["lemmas"])

            # If we are using BERT
            # Insert '[CLS]' and append '[SEP] 0 [SEP]'
            elif self.bert == True:
                copy_sent = list(sentence["lemmas"])
                copy_sent.insert(0, "[CLS]")
                copy_sent.append("[SEP]")
                copy_sent.append("<pad>")
                copy_sent.append("[SEP]")
                self.list_l_sentences.append(copy_sent)   

            self.list_l_unpacked_predicates.append(empty_predicate)
            self.list_l_pos.append(sentence['pos_tags'])                              

        # If sentene has 1 or more predicates
        elif num_pred > 0:

            for k in pred_idx:
                
                predicate = ["<pad>"] * len(sentence["lemmas"])

                if self.bert == False:
                    self.list_l_sentences.append(sentence["lemmas"])
                
                # If we are using BERT
                # Insert '[CLS]' and append '[SEP] k-th predicate [SEP]'                    
                elif self.bert == True:
                    copy_sent = list(sentence["lemmas"])
                    copy_sent.insert(0, "[CLS]")
                    copy_sent.append("[SEP]")
                    copy_sent.append(sentence["lemmas"][int(k)])
                    copy_sent.append("[SEP]")
                    self.list_l_sentences.append(copy_sent)    
                
                    predicate[int(k)] = sentence["predicates"][int(k)]
                
                self.list_l_pos.append(sentence['pos_tags'])   
                self.list_l_unpacked_predicates.append(predicate)                
        
        return
                

    def save_tokenizer(self, tokenizer_type, token_to_int, int_to_token):
        """
        Function that saves the input OR pos OR predicate OR label tokenizer

        Args: 
            tokenizer_type: "input" OR "pos" OR "predicate" OR "label"
            token_to_int: token to integer mapping dictionary to save
            int_to_token: integer to token mapping dictionary to save
        
        Returns: None        
        """ 
        if tokenizer_type == "input":
            token_to_int_path = self.word_to_int_path
            int_to_token_path = self.int_to_word_path
        elif tokenizer_type == "predicate":
            token_to_int_path = self.predicate_to_int_path
            int_to_token_path = self.int_to_predicate_path
        elif tokenizer_type == "pos":
            token_to_int_path = self.pos_to_int_path
            int_to_token_path = self.int_to_pos_path
        elif tokenizer_type == "label":
            token_to_int_path = self.label_to_int_path
            int_to_token_path = self.int_to_label_path

        with open(token_to_int_path, "wb") as handle:
            pickle.dump(token_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(int_to_token_path, "wb") as handle:
            pickle.dump(int_to_token, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return

    def create_tokenizer(self, tokenizer_type):
        """
        Function that creates the input OR POS tags OR predicate OR label tokenizer
        and saves them to disk

        Args: 
            tokenizer_type: "input" OR "pos" OR "predicate" OR "label"
        
        Returns: 
            token_to_int: dict that maps from 'token' to an integer        
            int_to_token: dict that maps from integer to 'token'   
            where 'token' is input words, POS tags, predicates, and labels  
        """ 

        tokens = list()
        token_to_int = OrderedDict()

        if tokenizer_type == "input":
            tokens = self.training_tokens
            dict_as_file = "./word_to_int.txt"
        elif tokenizer_type == "predicate":
            tokens = self.training_predicates
            dict_as_file = "./predicate_to_int.txt"
        elif tokenizer_type == "pos":
            tokens = self.training_pos
            dict_as_file = "./pos_to_int.txt"
        elif tokenizer_type == "label":
            tokens = self.training_labels
            dict_as_file = "./label_to_int.txt"     
        
        # Create token:int mapping
        token_to_int = OrderedDict({ni: indi for indi, ni in enumerate(set(tokens), start=2)})
        token_to_int["<pad>"] = 0
        token_to_int["<unk>"] = 1

        # Write the "token integer" mapping to file for debugging purposes
        sorted_keys = sorted(token_to_int, key=lambda k: token_to_int[k])
        with open(dict_as_file, "w", encoding="utf8") as f:
            for k in sorted_keys:
                f.write(k + " " + str(token_to_int[k]))
                f.write("\n")

        # create int:token mapping
        inv_d = {v: k for k, v in token_to_int.items()}  
        int_to_token = OrderedDict(sorted(inv_d.items(), key=lambda t: t[0])) 


        self.save_tokenizer(tokenizer_type, token_to_int, int_to_token)

        return token_to_int, int_to_token

    def create_all_tokenizers(self):
        """
        Function that creates the input AND POS Tags AND predicate AND label tokenizers

        Args: None
        
        Returns: 
            token_to_int: dict that maps from 'token' to an integer        
            int_to_token: dict that maps from integer to 'token'      
            where 'token' is input words, POS tags, predicates, and labels  
        """ 
        return self.create_tokenizer("input"), self.create_tokenizer("predicate"), self.create_tokenizer("pos"), self.create_tokenizer("label")


    def load_tokenizer(self, tokenizer_type):
        """
        Function that loads the input OR pos OR predicate OR label tokenizer

        Args: 
            tokenizer_type: "input" OR "pos" OR "predicate" OR "label"
        
        Returns: 
            token_to_int: dict that maps from token to and integer        
            int_to_token: dict that maps from integer to token        
        """ 
        if tokenizer_type == "input":
            token_to_int_path = self.word_to_int_path
            int_to_token_path = self.int_to_word_path
        elif tokenizer_type == "predicate":
            token_to_int_path = self.predicate_to_int_path
            int_to_token_path = self.int_to_predicate_path
        elif tokenizer_type == "pos":
            token_to_int_path = self.pos_to_int_path
            int_to_token_path = self.int_to_pos_path
        elif tokenizer_type == "label":
            token_to_int_path = self.label_to_int_path
            int_to_token_path = self.int_to_label_path

        with open(token_to_int_path, 'rb') as handle:
            token_to_int = pickle.load(handle)
        with open(int_to_token_path, 'rb') as handle:
            int_to_token = pickle.load(handle)   
        
        return token_to_int, int_to_token

    def load_all_tokenizers(self):
        """
        Function that loads the input AND POS AND predciate AND label tokenizers from disk

        Args: None
        
        Returns: 
            token_to_int: dict that maps from 'token' to and integer        
            int_to_token: dict that maps from integer to 'token'      
            where 'token' is input words, POS tags, predicates, and labels    
        """ 
        return self.load_tokenizer("input"), self.load_tokenizer("predicate"), self.load_tokenizer("pos"), self.load_tokenizer("label")

    def encode_text(self, text_type):
        """
        Function that encods input words OR POS tags OR predicates OR labels to torch.tensors

        Args: 
            text_type: "input" OR "pos" OR "predicate" OR "label"

        Returns: 
            data: list of torchTensors representing the encoded data
        """ 
        
        token_to_int = dict()
        list_l_tokens = list()

        if text_type == "input":
            token_to_int, _ = self.load_tokenizer("input")
            list_l_tokens = self.list_l_sentences

        elif text_type == "predicate":
            token_to_int, _ = self.load_tokenizer("predicate")
            list_l_tokens = self.list_l_unpacked_predicates

        elif text_type == "pos":
            token_to_int, _ = self.load_tokenizer("pos")
            list_l_tokens = self.list_l_pos

        elif text_type == "label":
            token_to_int, _ = self.load_tokenizer("label")
            list_l_tokens = self.list_l_labels

        # data is the text converted to indexes, as list of lists
        data = []

        if (text_type == "input" and self.bert == False) or text_type != "input":            

            # for each sentence
            for sentence in list_l_tokens:
                paragraph = []
                # for each token in the sentence, map it to an int 
                # using its corresponding tokenizer
                for i in sentence:
                    id_ = token_to_int[i] if i in token_to_int else token_to_int["<unk>"]
                    paragraph.append(id_)

                paragraph = torch.LongTensor(paragraph)
                data.append(paragraph)
        
        # If we are using the BERT model, use the BERT Tokenizer
        elif text_type == "input" and self.bert == True:

            bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        
            for sentence in list_l_tokens:
                encoding = bert_tokenizer.encode_plus(
                sentence,
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_tensors='pt',  # Return PyTorch tensors
                )
                data.append(torch.squeeze(encoding['input_ids']))

        return data

    def encode_all_data(self):
        """
        Function that encods input words, POS tags, predicates, and labels to torch.tensors

        Args: 
        
        Returns: 
            tuple of lists of torchTensors representing the encoded data
        """ 
        
        # If we are NOT running "implementation.py"
        # we can include or NOT include POS Tags data
        if self.mode == "train" or self.mode == "dev" or self.mode == "test":
            if self.include_pos == True:
                return self.encode_text("input"), self.encode_text("predicate"), self.encode_text("pos"), self.encode_text("label") 
            elif self.include_pos == False:
                return self.encode_text("input"), self.encode_text("predicate"), None, self.encode_text("label") 
        
        # If we are running "implementation.py", there are no labels
        # and again we can include or NOT include POS Tags data    
        # 
        # We return "None" in place of any non-existant POS tag or label data
        # this allows the dataloader class to create a batch that includes the correct elements
        # (inputs only...inputs and pos only...inputs,pos, and labels, and so on)    
        elif self.mode == "submit":
            if self.include_pos == True:
                return self.encode_text("input"), self.encode_text("predicate"), self.encode_text("pos"), None 
            elif self.include_pos == False:
                return self.encode_text("input"), self.encode_text("predicate"), None, None
