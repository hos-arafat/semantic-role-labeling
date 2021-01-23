
class HParams():
    """Class that specifies the model's hyperparameters."""
    def __init__(self, vocabulary, pred_vocabulary, pos_vocabulary, label_vocabulary, opts):
        """
        Args:
            vocabulary: dictionary mapping input words to ints
            pred_vocabulary: dictionary mapping predicates to ints
            pos_vocabulary: dictionary mapping POS tags to ints
            label_vocabulary: dictionary mapping ground truth labels to ints
            opts: dictionary that specifies various training options
        """ 

        self.vocab_size = len(vocabulary)

        self.use_binary_predicate = opts["use_binary_pred"]

        if opts["task"] == "34" and pred_vocabulary is not None:
            self.pred_vocab_size = len(pred_vocabulary)
            self.use_pred_embeddings = opts['use_pred_embeddings']

        self.label_vocabulary = len(label_vocabulary)
        self.pos_vocab_size = len(pos_vocabulary)
        self.num_classes = len(label_vocabulary)

        self.hidden_dim = opts["hidden_dim"]
        self.embedding_dim = opts['glove_embedding_dim']
        self.pred_embedding_dim = opts['pred_embedding_dim']
        self.pos_embedding_dim = opts['pos_embedding_dim']

        self.use_pos_embeddings = opts['use_pos_embeddings']

        self.bidirectional = opts['bidirectional']
        self.num_layers = opts['lstm_layers']
        self.dropout = opts['dropout']

        self.embeddings = None
        self.pred_embeddings = None
        self.pos_embeddings = None