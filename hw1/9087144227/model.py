import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

        # #load glove vocab
        # if args.emb_file is not None:
        #     self.copy_embedding_from_numpy()


    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path, weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    print(f"Loading from {emb_file}")

    vocab_size = len(vocab)

    emb = np.random.uniform(
        low=-0.05, high=0.05,
        size=(vocab_size, emb_size)
    ).astype(np.float32)

    hit = 0

    with open(emb_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]

            if word in vocab.word2id:  # only keep vocab words
                vec = np.asarray(parts[1:], dtype=np.float32)
                emb[vocab[word]] = vec
                hit += 1

    print(f"Embedding coverage: {hit}/{vocab_size} = {hit/vocab_size:.2%}")

    return emb



    pass
    #raise NotImplementedError()


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        #from args
        self.emb_size = int(self.args.emb_size)
        self.hid_size = int(self.args.hid_size)
        self.pooling_method = self.args.pooling_method
        
        vocab_size = len(self.vocab)
        pad_id = self.vocab["<pad>"]


        #layers
        self.embedding = nn.Embedding(vocab_size, self.emb_size, padding_idx=pad_id)
        self.fc1 = nn.Linear(self.emb_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.tag_size)
        self.relu = nn.ReLU()

        self.pad_id = pad_id 

        #raise NotImplementedError()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



        #raise NotImplementedError()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb = load_embedding(
            self.vocab,
            self.args.emb_file,
            self.emb_size
        )

        self.embedding.weight.data.copy_(
            torch.from_numpy(emb)
        )

        
        #raise NotImplementedError()

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        emb = self.embedding(x)

        # add mask 

        mask = (x )

        if self.pooling_method == 'avg':
            x_agg = emb.mean(dim=1)

        h = self.relu(self.fc1(x_agg))
        out = self.fc2(h)

        return out
        #raise NotImplementedError()
