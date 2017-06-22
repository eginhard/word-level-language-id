# -*- coding: utf-8 -*-

import codecs
import json
import math

class LanguageModel:
    """Language model based on a lexicon and a character n-gram Markov model.
    """

    # Symbols to mark start and end of a sequence.
    START = "<"
    END = ">"
    # Identifier for unseen ngrams
    UNKNOWN = "UNKNOWN"
    # Identifier for out-of-lexicon words
    OOV = "###"

    # Case for words shorter than this will be preserved in the lexicon.
    # Idea is that shorter words will be frequent enough to warrant this
    # and case is especially important for distinguishing English "I".
    # Helps with certain sentence-initial patterns as well, e.g.
    # "is" and "a" are frequent in both languages within a sentence,
    # but "Is" will rather start an Irish and "A" an English one.
    CASE_SENSITIVITY_THRESHOLD = 4

    def __init__(self, language, n, lex_file, lex_weight=1):
        """Initialises the language model.

        Args:
            language (str): Name of the language for this model.
            n (int): Ngram length.
            lex_file (str): Frequency lexicon file name.
            lex_weight (float): Weight of the lexicon vs. the character model.
        """
        self.language = language
        self.n = n
        self.lex_file = lex_file

        self.lex_weight = lex_weight
        self.char_weight = 1 - lex_weight

        # Probabilities of how often each ngram occurs at the start of a token.
        self.start_prob = {}
        # Probabilities of transitions between ngrams.
        # This is a nested dictionary, e.g. trans_prob["bai"]["ail"]
        self.trans_prob = {}
        # Stores relative lexicon frequency of every word.
        self.lex = self.load_lexicon(lex_file)

    def dump(self, file_name=None):
        """Saves the language model to the specified file in JSON format"""
        if file_name is None:
            file_name = self.language + ".model"
        with open(file_name, "w") as f:
            json.dump([self.language, self.n,
                       self.start_prob, self.trans_prob], f)
        print("Saved model at: %s" % file_name)

    @classmethod
    def load(cls, model_file, lex_file, lex_weight=1):
        """Loads the language model from the specified file

        Args:
            model_file (str): LanguageModel file name
            lex_file (str): Frequency lexicon file name
            lex_weight (float): Weight of the lexicon vs. the character model.
        """
        with open(model_file) as f:
            language, n, start_prob, trans_prob = json.load(f)
            model = cls(language, n, lex_file, lex_weight)
            model.start_prob = start_prob
            model.trans_prob = trans_prob
            return model

    @classmethod
    def word2ngrams(cls, word, n):
        """Splits a word into a list of character ngrams, adding start and end symbols.
        word2ngram("agus", 3) = ["<ag", "agu", "gus", "us>"]
        """
        word = cls.START + word + cls.END
        if n >= 4:
            word = cls.START + word + cls.END
        return [word[i:i+n] for i in range(len(word)-n+1)]

    def load_lexicon(self, lex_file):
        """Loads the frequency lexicon into the language model.
        """
        lamb = 0.1 # smoothing value

        with codecs.open(lex_file, encoding='utf-8') as f:
            lex = {}
            total = 0
            for line in f:
                fields = line.strip().split()
                word = fields[0]
                if len(fields[0]) >= self.CASE_SENSITIVITY_THRESHOLD:
                    word = fields[0].lower()
                lex[word] = lex.get(word, lamb) + int(fields[1])
                total += int(fields[1])
            # Add out-of-lexicon word
            lex[self.OOV] = lamb

            for word in lex:
                lex[word] = math.log(lex[word] / float(total + lamb * (len(lex))))
            return lex
        return None

    def lex_score(self, word):
        """Returns the log probability of the given word according to the lexicon.
        """
        if len(word) >= self.CASE_SENSITIVITY_THRESHOLD:
                word = word.lower()
        return self.lex.get(word, self.lex[self.OOV])

    def char_score(self, word, debug=False):
        """Returns the log probability of the given word according to the character model.

        Enabling <debug> allows inspecting individual transition probabilities.
        """
        ngrams = self.word2ngrams(word, self.n)
        logp = 0
        # Add starting probability
        if not ngrams[0] in self.start_prob.keys():
            logp += self.start_prob[self.UNKNOWN]
        else:
            logp += self.start_prob[ngrams[0]]
        debugstr = word + " " + str(logp)
        # Add transition probabilities
        for i in range(len(ngrams)-1):
            if not ngrams[i] in self.trans_prob.keys():
                logp += self.trans_prob[self.UNKNOWN][self.UNKNOWN]
                debugstr += " " + ngrams[i] + " XX" + str(self.trans_prob[self.UNKNOWN][self.UNKNOWN])
            elif not ngrams[i+1] in self.trans_prob[ngrams[i]].keys():
                logp += self.trans_prob[ngrams[i]][self.UNKNOWN]
                debugstr += " " + ngrams[i] + " X" + str(self.trans_prob[ngrams[i]][self.UNKNOWN])
            else:
                logp += self.trans_prob[ngrams[i]][ngrams[i+1]]
                debugstr += " " + ngrams[i] + " " + str(self.trans_prob[ngrams[i]][ngrams[i+1]])
        if debug:
            print(debugstr)
        return logp

    def train(self, smooth_lambda=0.001):
        """Trains a character ngram model on the specified corpus.

        Args:
            smooth_lambda (float): lambda value for add-lambda smoothing
        """
        lamb = smooth_lambda

        # Total number of tokens.
        token_total = 0
        # Counts of how often each ngram occurs at the start of a token.
        start_count = {}
        # Counts of transitions between ngrams.
        # This is a nested dictionary, e.g. trans_count["bai"]["ail"]
        trans_count = {}
        # Total number of transitions from each ngram (i.e. ngram count excluding
        # sequence-final ngrams).
        trans_total = {}
        # Set of all characters.
        charset = set()

        n = self.n
        print("Training %d-gram model for language: %s" % (n, self.language))

        # Calculate counts
        with codecs.open(self.lex_file, encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split()
                token = self.START + fields[0] + self.END
                token_count = int(fields[1])
                token_total += token_count
                charset.update(list(token))
                # Increase start counter
                start_count[token[:n]] = start_count.get(token[:n], lamb) + token_count
                # Increase transition counter
                for i in range(len(token) - n):
                    if not token[i:i+n] in trans_count.keys():
                        trans_count[token[i:i+n]] = {}
                    trans_count[token[i:i+n]][token[i+1:i+n+1]] = trans_count[token[i:i+n]].get(token[i+1:i+n+1], lamb) + token_count
                    trans_total[token[i:i+n]] = trans_total.get(token[i:i+n], 0) + token_count

        # Smoothing for unseen ngrams
        self.trans_prob[self.UNKNOWN] = {}
        self.trans_prob[self.UNKNOWN][self.UNKNOWN] = math.log(1 / float(len(charset) + 1))

        # Calculate starting probabilities
        denominator = token_total + lamb * (len(start_count) + 1)
        for ngram in start_count.keys():
            self.start_prob[ngram] = math.log(start_count[ngram] / denominator)
        self.start_prob[self.UNKNOWN] = math.log(lamb / denominator)

        # Calculate transition probabilities
        for ngram in trans_count.keys():
            self.trans_prob[ngram] = {}
            denominator = (trans_total[ngram]) + lamb * (len(trans_count[ngram]) + 1)
            for next_ngram in trans_count[ngram].keys():
                self.trans_prob[ngram][next_ngram] = math.log(
                    trans_count[ngram][next_ngram] / denominator)
            self.trans_prob[ngram][self.UNKNOWN] = math.log(lamb / denominator)

        print("Model trained on %d tokens" % token_total)
