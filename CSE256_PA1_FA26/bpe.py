# bpe.py
# Byte Pair Encoding (Sennrich et al., 2016) for CSE 256 PA1, Part 2.
#
# Training: start from characters (+ an end-of-word marker) and repeatedly merge the most
# frequent adjacent pair of symbols in the training corpus until the vocabulary reaches the
# requested size. Encoding: split a word into characters and replay the learned merges in
# the order they were learned.

from collections import Counter, defaultdict
from utils import Indexer
from sentiment_data import read_sentiment_examples

END_OF_WORD = '</w>'   # marks the last symbol of a word, so "er</w>" (better) != "er" (very)
PAD_TOKEN, UNK_TOKEN = 'PAD', 'UNK'


class BPETokenizer:
    def __init__(self, vocab_size, lowercase=True):
        """
        :param vocab_size: target number of subword types (base characters + merged symbols),
                           not counting PAD and UNK
        :param lowercase: lowercase text before learning/applying merges
        """
        self.target_vocab_size = vocab_size
        self.lowercase = lowercase
        self.merges = []          # list of (left, right) pairs, in the order they were learned
        self.merge_ranks = {}     # (left, right) -> position in self.merges
        self.vocab = []           # subword types: base alphabet followed by merged symbols
        self.indexer = None       # Indexer over PAD, UNK and self.vocab
        self.cache = {}           # word -> list of subwords (words repeat a lot)

    # ------------------------------------------------------------------ training
    def train(self, sentences):
        """
        Learn merges from a list of tokenized sentences (each a list of words).
        Pair counts are kept incrementally: after each merge only the words that contain the
        merged pair are re-counted, which keeps training to a few seconds.
        """
        word_freqs = Counter(self._norm(w) for sent in sentences for w in sent)
        words = [list(w) + [END_OF_WORD] for w in word_freqs]   # symbols of each unique word
        freqs = list(word_freqs.values())

        alphabet = sorted({s for w in words for s in w})
        num_merges = max(0, self.target_vocab_size - len(alphabet))

        # Initial pair counts (weighted by word frequency) and pair -> words index
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for i, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[i]
                pair_words[pair].add(i)

        merged_symbols = []
        for _ in range(num_merges):
            if not pair_counts:
                break                                   # every word is already one symbol
            # Most frequent pair (ties broken alphabetically so results are deterministic)
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < 1:
                break
            new_symbol = best[0] + best[1]
            self.merges.append(best)
            merged_symbols.append(new_symbol)

            for i in list(pair_words[best]):
                w, f = words[i], freqs[i]
                # remove this word's old pair counts
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_words[pair].discard(i)
                # merge every occurrence of the best pair
                w = self._merge_pair(w, best, new_symbol)
                words[i] = w
                # add the new pair counts
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    pair_words[pair].add(i)
            pair_words.pop(best, None)

        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}
        self.vocab = alphabet + merged_symbols
        self.indexer = Indexer()
        self.indexer.add_and_get_index(PAD_TOKEN)      # index 0
        self.indexer.add_and_get_index(UNK_TOKEN)      # index 1
        for sym in self.vocab:
            self.indexer.add_and_get_index(sym)
        self.cache = {}
        return self

    def train_from_file(self, infile):
        return self.train([ex.words for ex in read_sentiment_examples(infile)])

    # ------------------------------------------------------------------ encoding
    def encode_word(self, word):
        """Split one word into subwords by replaying the learned merges in order."""
        word = self._norm(word)
        if word in self.cache:
            return self.cache[word]
        symbols = list(word) + [END_OF_WORD]
        while len(symbols) > 1:
            # apply the earliest-learned merge that is present in the word
            ranked = [(self.merge_ranks.get(p, float('inf')), p) for p in zip(symbols, symbols[1:])]
            rank, pair = min(ranked)
            if rank == float('inf'):
                break
            symbols = self._merge_pair(symbols, pair, pair[0] + pair[1])
        self.cache[word] = symbols
        return symbols

    def encode(self, words):
        """Tokenize a sentence (list of words) into a flat list of subwords."""
        return [sub for w in words for sub in self.encode_word(w)]

    def encode_ids(self, words):
        """Tokenize a sentence into subword indices; unseen characters map to UNK."""
        ids = []
        for sub in self.encode(words):
            idx = self.indexer.index_of(sub)
            ids.append(idx if idx != -1 else self.indexer.index_of(UNK_TOKEN))
        return ids

    # ------------------------------------------------------------------ helpers
    def vocab_len(self):
        """Embedding table size: subword types + PAD + UNK."""
        return len(self.indexer)

    def _norm(self, word):
        return word.lower() if self.lowercase else word

    @staticmethod
    def _merge_pair(symbols, pair, new_symbol):
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(new_symbol)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out
