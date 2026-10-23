# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from sentiment_data import read_word_embeddings
from DANmodels import SentimentDatasetDAN, SentimentDatasetSubwordDAN, DAN, INIT_SCHEMES
from bpe import BPETokenizer


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
# Default learning rate for each optimizer (used when --lr is not given).
# SGD and Adagrad need much larger steps than the Adam family.
DEFAULT_LR = {'adam': 1e-3, 'adamw': 1e-3, 'rmsprop': 1e-3, 'adagrad': 1e-2, 'sgd': 5e-2}


# Build the optimizer chosen on the command line
def get_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        # AdamW applies weight decay directly to the weights (decoupled from the gradient)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay if weight_decay > 0 else 0.01)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def experiment(model, train_loader, test_loader, lr=0.0001, epochs=100, print_every=10,
               optimizer_name='adam', weight_decay=0.0, momentum=0.9):
    loss_fn = nn.NLLLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay, momentum)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % print_every == print_every - 1:
            print(f'Epoch #{epoch + 1}: train loss {train_loss:.3f}, dev loss {test_loss:.3f}, '
                  f'train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')

    return all_train_accuracy, all_test_accuracy


# Plot train/dev accuracy curves for one model and save to file
def plot_curves(train_acc, dev_acc, title, outfile):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='train')
    plt.plot(range(1, len(dev_acc) + 1), dev_acc, label='dev')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(outfile)
    print(f"Accuracy plot saved as {outfile}")


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (BOW, DAN or SUBWORDDAN)')
    # DAN hyperparameters (defaults = best configuration found)
    parser.add_argument('--emb', type=str, default='300', choices=['50', '300'], help='GloVe embedding dimension')
    parser.add_argument('--hidden', type=int, default=300, help='Hidden layer size')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout after each hidden layer')
    parser.add_argument('--emb_dropout', type=float, default=0.0, help='Dropout on the word embeddings (before averaging)')
    parser.add_argument('--word_dropout', type=float, default=0.3, help='Probability of dropping a word before averaging')
    parser.add_argument('--freeze', action='store_true', help='Keep GloVe embeddings frozen')
    parser.add_argument('--random_init', action='store_true',
                        help='Part 1b: randomly initialise the embeddings instead of using GloVe')
    parser.add_argument('--init', type=str, default='default', choices=INIT_SCHEMES,
                        help='Weight initialisation for the hidden/output Linear layers')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu'],
                        help='Nonlinearity between hidden layers')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd', 'adagrad', 'rmsprop'], help='Optimizer for the DAN')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default depends on optimizer: 1e-3 Adam/AdamW/RMSprop, 1e-2 Adagrad, 5e-2 SGD)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay (AdamW uses 0.01 if not set)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (SGD only)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: 10 for DAN, 20 for SUBWORDDAN)')
    # Part 2: subword (BPE) DAN
    parser.add_argument('--vocab_size', type=int, default=5000,
                        help='SUBWORDDAN: BPE vocabulary size (base characters + merges)')
    parser.add_argument('--subword_emb_dim', type=int, default=300,
                        help='SUBWORDDAN: dimension of the randomly initialised subword embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Parse the command-line arguments
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 20 if args.model == 'SUBWORDDAN' else 10
    if args.lr is None:
        args.lr = DEFAULT_LR[args.optimizer]
    torch.manual_seed(args.seed)

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Load GloVe embeddings and index the data with their vocabulary
        start_time = time.time()
        word_embeddings = read_word_embeddings(f"data/glove.6B.{args.emb}d-relativized.txt")
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        print(f"Data loaded in : {time.time() - start_time:.1f} seconds")

        model = DAN(word_embeddings, hidden_size=args.hidden, num_layers=args.layers,
                    dropout=args.dropout, word_dropout=args.word_dropout, freeze=args.freeze,
                    activation=args.activation, emb_dropout=args.emb_dropout,
                    init=args.init, random_init=args.random_init)
        emb_source = 'random-init' if args.random_init else 'GloVe'
        print(f'\nDAN ({args.emb}d {emb_source}, {args.layers} hidden layers of {args.hidden}, '
              f'emb dropout {args.emb_dropout}, hidden dropout {args.dropout}, word dropout {args.word_dropout}, '
              f'activation {args.activation}, init {args.init}, optimizer {args.optimizer}, lr {args.lr}):')

        start_time = time.time()
        dan_train_accuracy, dan_test_accuracy = experiment(model, train_loader, test_loader,
                                                           lr=args.lr, epochs=args.epochs, print_every=1,
                                                           optimizer_name=args.optimizer,
                                                           weight_decay=args.weight_decay,
                                                           momentum=args.momentum)
        best_epoch = max(range(len(dan_test_accuracy)), key=lambda i: dan_test_accuracy[i])
        print(f"\nBest dev accuracy {dan_test_accuracy[best_epoch]:.3f} at epoch {best_epoch + 1}; "
              f"final dev accuracy {dan_test_accuracy[-1]:.3f}")
        print(f"Training time: {time.time() - start_time:.1f} seconds\n")

        plot_curves(dan_train_accuracy, dan_test_accuracy,
                    f'DAN ({args.emb}d {emb_source}, {args.activation}, {args.optimizer}) Train/Dev Accuracy',
                    f'dan_accuracy_{"random" if args.random_init else "glove"}_{args.activation}_{args.optimizer}.png')
    elif args.model == "SUBWORDDAN":
        # 1. Learn BPE merges on the training set only
        start_time = time.time()
        bpe = BPETokenizer(args.vocab_size).train_from_file("data/train.txt")
        bpe_time = time.time() - start_time
        print(f"BPE trained in {bpe_time:.1f} seconds: {len(bpe.vocab)} subword types "
              f"({len(bpe.merges)} merges) + PAD/UNK")
        example = "This unwatchable movie is surprisingly heartfelt .".split()
        print(f"Example tokenization: {bpe.encode(example)}")

        # 2. Encode train/dev into padded subword index tensors
        start_time = time.time()
        train_data = SentimentDatasetSubwordDAN("data/train.txt", bpe)
        dev_data = SentimentDatasetSubwordDAN("data/dev.txt", bpe)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        print(f"Data encoded in {time.time() - start_time:.1f} seconds; average tokens per sentence: "
              f"train {train_data.avg_tokens:.1f}, dev {dev_data.avg_tokens:.1f}")

        # 3. Same DAN, but with randomly initialised subword embeddings
        model = DAN(None, vocab_size=bpe.vocab_len(), emb_dim=args.subword_emb_dim,
                    hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout,
                    word_dropout=args.word_dropout, activation=args.activation,
                    emb_dropout=args.emb_dropout, init=args.init)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'\nSubword DAN (BPE vocab {args.vocab_size}, {args.subword_emb_dim}d random-init, '
              f'{args.layers} hidden layers of {args.hidden}, emb dropout {args.emb_dropout}, '
              f'hidden dropout {args.dropout}, token dropout {args.word_dropout}, activation {args.activation}, '
              f'init {args.init}, optimizer {args.optimizer}, lr {args.lr}, {num_params:,} parameters):')

        start_time = time.time()
        sub_train_accuracy, sub_test_accuracy = experiment(model, train_loader, test_loader,
                                                           lr=args.lr, epochs=args.epochs, print_every=1,
                                                           optimizer_name=args.optimizer,
                                                           weight_decay=args.weight_decay,
                                                           momentum=args.momentum)
        train_time = time.time() - start_time
        best_epoch = max(range(len(sub_test_accuracy)), key=lambda i: sub_test_accuracy[i])
        print(f"\nBest dev accuracy {sub_test_accuracy[best_epoch]:.3f} at epoch {best_epoch + 1}; "
              f"final dev accuracy {sub_test_accuracy[-1]:.3f}")
        print(f"Training time: {train_time:.1f} seconds ({train_time / args.epochs:.1f} s/epoch); "
              f"BPE training time: {bpe_time:.1f} seconds\n")

        plot_curves(sub_train_accuracy, sub_test_accuracy,
                    f'Subword DAN (BPE vocab {args.vocab_size}) Train/Dev Accuracy',
                    f'subword_dan_accuracy_v{args.vocab_size}.png')

    else:
        print(f"Unknown model type: {args.model}. Use BOW, DAN or SUBWORDDAN.")


if __name__ == "__main__":
    main()
