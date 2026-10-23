import torch
from torch import nn
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract labels from the examples
        self.labels = [ex.label for ex in self.examples]

        indexer = word_embeddings.word_indexer
        index_lists = []

        for ex in self.examples:
            idxs = []
            for word in ex.words:
                word = word.lower()                # GloVe vocab is lowercase
                idx = indexer.index_of(word)
                if idx == -1:                      # still not found -> UNK
                    idx = 1
                idxs.append(idx)
            index_lists.append(idxs)

        # Pad every sentence with PAD (0) to the same length
        max_len = max(len(idxs) for idxs in index_lists)
        padded = [idxs + [0] * (max_len - len(idxs)) for idxs in index_lists]

        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(padded, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the token index sequence and label for the given index
        return self.embeddings[idx], self.labels[idx]



class SentimentDatasetSubwordDAN(Dataset):
    """
    Part 2: each sentence becomes a padded tensor of BPE subword indices.
    The tokenizer must already be trained (on train.txt only).
    """
    def __init__(self, infile, bpe):
        self.examples = read_sentiment_examples(infile)
        index_lists = [bpe.encode_ids(ex.words) for ex in self.examples]

        # Pad every sentence with PAD (0) to the same length
        max_len = max(len(idxs) for idxs in index_lists)
        padded = [idxs + [0] * (max_len - len(idxs)) for idxs in index_lists]

        self.embeddings = torch.tensor(padded, dtype=torch.long)
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)
        self.avg_tokens = sum(len(idxs) for idxs in index_lists) / len(index_lists)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Nonlinearities that can be used between the hidden layers
ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
}


# Weight initialisation schemes for the Linear layers (embeddings stay GloVe-initialised)
def init_linear(layer, scheme):
    if scheme == 'default':           # PyTorch default: Kaiming-uniform with a=sqrt(5)
        return
    if scheme == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight)
    elif scheme == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif scheme == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    elif scheme == 'kaiming_normal':
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    elif scheme == 'orthogonal':
        nn.init.orthogonal_(layer.weight)
    elif scheme == 'small_normal':    # N(0, 0.01^2): very small weights
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    elif scheme == 'zeros':           # all weights 0: shows why symmetry must be broken
        nn.init.zeros_(layer.weight)
    else:
        raise ValueError(f"Unknown init scheme: {scheme}")
    nn.init.zeros_(layer.bias)


INIT_SCHEMES = ['default', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                'kaiming_normal', 'orthogonal', 'small_normal', 'zeros']


class DAN(nn.Module):
    """
    Deep Averaging Network (Iyyer et al., 2015):
    embed tokens -> (word dropout) -> (embedding dropout) -> masked average
    -> [Linear -> activation -> hidden Dropout] x num_layers
    -> Linear -> LogSoftmax
    """
    def __init__(self, word_embeddings, hidden_size=300, num_layers=2, dropout=0.2,
                 word_dropout=0.3, freeze=False, activation='relu', emb_dropout=0.0,
                 init='default', random_init=False, vocab_size=None, emb_dim=None):
        """
        word_embeddings: GloVe WordEmbeddings for word-level models (Parts 1a/1b).
                         Pass None for the subword model (Part 2) and give vocab_size and
                         emb_dim instead; embeddings are then always randomly initialised.
        """
        super().__init__()
        self.pad_idx = 0
        self.word_dropout = word_dropout

        if word_embeddings is not None:
            vocab_size = len(word_embeddings.word_indexer)
            emb_dim = word_embeddings.get_embedding_length()
        else:
            # Part 2: BPE subwords have no pretrained vectors
            random_init = True

        if random_init:
            # Parts 1b and 2: randomly initialised embeddings, learned from scratch.
            # For 1b the GloVe file is used only for its vocabulary (word -> index) and dimension;
            # none of its vectors are used. nn.Embedding draws each vector from N(0, 1),
            # and the PAD row (index 0) is fixed at zero and never updated.
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=self.pad_idx)
        else:
            # Part 1a: embedding layer initialised with pretrained GloVe vectors (50d or 300d)
            self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=freeze)

        # Dropout applied element-wise to the word embeddings, before averaging
        self.emb_dropout = nn.Dropout(emb_dropout)

        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_size), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)

        self.out = nn.Linear(in_dim, 2)

        # Initialise every Linear layer (hidden + output) with the chosen scheme
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, init)

    def forward(self, token_ids):
        # main.py casts inputs with X.float(); nn.Embedding needs integer indices
        token_ids = token_ids.long()
        # 1. Mask of real tokens
        mask = (token_ids != self.pad_idx)                      # (B, T)
        # 2. Word dropout (training only): drop whole tokens
        if self.training and self.word_dropout > 0:
            keep = torch.rand_like(token_ids, dtype=torch.float) > self.word_dropout
            new_mask = mask & keep
            # Don't let any example lose all its tokens
            empty = new_mask.sum(dim=1) == 0
            new_mask[empty] = mask[empty]
            mask = new_mask
        # 3. Embed
        x = self.embedding(token_ids)                           # (B, T, d)
        x = self.emb_dropout(x)                                 # no-op when emb_dropout=0 or in eval mode
        # 4. Masked average
        mask_f = mask.unsqueeze(-1).float()                     # (B, T, 1)
        summed = (x * mask_f).sum(dim=1)                        # (B, d)
        counts = mask_f.sum(dim=1).clamp(min=1)                 # (B, 1)
        avg = summed / counts                                   # (B, d)
        # 5. Deep feed-forward layers
        z = self.mlp(avg)                                       # (B, h)
        # 6. Output: log-probabilities for NLLLoss
        logits = self.out(z)                                    # (B, 2)
        return torch.log_softmax(logits, dim=1)
