# -*- coding: utf-8 -*-

import json
import math
import sys

from LanguageModel import LanguageModel

class LanguageIdentifier:
    """Word level language identification.
    """

    GA = "ga"
    EN = "en"

    # This string should be used for tokens that don't need a language
    # assignment, e.g. punctuation.
    IGNORE = "##IGNORE##"

    def __init__(self,
                 model_file_ga, model_file_en,
                 lex_file_ga, lex_file_en,
                 lex_weight=1):
        """Initialises the language model.

        Args:
            model_file_ga/en (str): Irish/English LanguageModel file name
            lex_file_ga/en (str): Irish/English Lexicon (1 word + frequency per line)
            lex_weight (float): Weight of the lexicon vs. the character model.
        """
        self.lex_weight = lex_weight
        self.model = {}
        self.model[self.GA] = LanguageModel.load(model_file_ga, lex_file_ga, lex_weight)
        self.model[self.EN] = LanguageModel.load(model_file_en, lex_file_en, lex_weight)

    def identify(self, tokens,
                 method="viterbi",
                 transition_probability=0.78,
                 start_probability=0.75):
        """Word level language identification of a token sequence.

        Args:
            tokens (str[]): A list of tokens
            method (str): One of [independent, viterbi]

            VITERBI
            transition_probability (float):
                Probability that the following token will be in the same language.
            start_probability (float):
                Probability that the first token will be Irish

        Returns:
            languages (str[]): List of language assignments matching the token list.
                               E.g. ["ga", "ga", "en", ...]
        """

        # Special treatment for the Irish affirmative "sea" in 1-word sentences.
        if len(tokens) == 1 and tokens[0].lower() == "sea":
            return [self.GA]

        if method == "independent":
            return self.identify_independent(tokens)
        elif method == "viterbi":
            return self.identify_viterbi(tokens,
                                         transition_probability,
                                         start_probability)

    def identify_independent(self, tokens):
        """Independent word level language identification without any context.

        For each token, picks the language with the higher probability according
        to the language models.
        """

        languages = []
        for token in tokens:
            scores = self.score(token)
            if scores[self.GA] >= scores[self.EN]:
                languages.append(self.GA)
            else:
                languages.append(self.EN)

        return languages

    def identify_viterbi(self, tokens,
                         transition_probability=0.9,
                         start_probability=0.5):
        """Context-dependent word level language identification using Viterbi.

        Assigns the most likely language to each token according to both language
        models and the likelihood of switching the language or not.
        """

        languages = []

        V = [{}] # Stores max probability for each token and language
        S = [{}] # Stores argmax (most likely language)

        # Probability of keeping vs. switching the language
        trans_p = {}
        trans_p[self.GA] = {}
        trans_p[self.EN] = {}
        trans_p[self.GA][self.GA] = transition_probability
        trans_p[self.EN][self.EN] = transition_probability
        trans_p[self.GA][self.EN] = 1- transition_probability
        trans_p[self.EN][self.GA] = 1- transition_probability

        # Initial probabilities for both languages
        scores = self.score(tokens[0])
        V[0][self.GA] = math.log(start_probability) + scores[self.GA]
        V[0][self.EN] = math.log(1 - start_probability) + scores[self.EN]

        langs = [self.GA, self.EN]

        # Iterate over tokens (starting at second token)
        for t in range(1, len(tokens)):
            V.append({})
            S.append({})
            # Iterate over the two languages
            scores = self.score(tokens[t])
            for lang in langs:
                # Get max and argmax for current position
                term = (V[t-1][lang2] + math.log(trans_p[lang2][lang]) + scores[lang]
                        for lang2 in langs)
                maxlang, prob = self.max_argmax(term)
                V[t][lang] = prob
                S[t][lang] = langs[maxlang]

        # Get argmax for final token
        languages = [0] * len(tokens)
        languages[-1] = langs[self.max_argmax(V[-1][lang] for lang in langs)[0]]

        # Reconstruct optimal path
        for t in range(len(tokens)-1, 0, -1):
            languages[t-1] = S[t][languages[t]]

        return languages

    def max_argmax(self, iterable):
        """Returns the tuple (argmax, max) for a list
        """
        return max(enumerate(iterable), key=lambda x: x[1])

    def score(self, word):
        """Returns the weighted log probability according to lexicon + character model.
        """
        # Punctuation etc. have no influence on the language assignment
        if word == self.IGNORE:
            return {self.GA: 1, self.EN: 1}

        lex_score, char_score = {}, {}
        for lang in [self.GA, self.EN]:
            lex_score[lang] = math.exp(self.model[lang].lex_score(word))
            char_score[lang] = math.exp(self.model[lang].char_score(word))

        # Relative scores, only these can be weighted
        lex_score_rel, char_score_rel = {}, {}
        for lang in [self.GA, self.EN]:
            lex_score_rel[lang] = lex_score[lang] / (lex_score[self.GA] +
                                                     lex_score[self.EN])
            char_score_rel[lang] = char_score[lang] / (char_score[self.GA] +
                                                       char_score[self.EN])

        weighted_score = {}
        # If neither word is in the lexicon, use only the character model
        if (lex_score[self.GA] == math.exp(self.model[self.GA].lex_score(LanguageModel.OOV)) and
            lex_score[self.EN] == math.exp(self.model[self.EN].lex_score(LanguageModel.OOV))):
            for lang in [self.GA, self.EN]:
                weighted_score[lang] = math.log(char_score_rel[lang])
        # Else combine both models
        else:
            for lang in [self.GA, self.EN]:
                weighted_score[lang] = math.log(self.lex_weight * lex_score_rel[lang] +
                                                (1 - self.lex_weight) * char_score_rel[lang])
        #print word
        #print("%.15f %.15f" % (lex_score[self.GA], lex_score[self.EN]))
        #print("%.15f %.15f" % (char_score[self.GA], char_score[self.EN]))
        #print("%.15f %.15f" % (lex_score_rel[self.GA], lex_score_rel[self.EN]))
        #print("%.15f %.15f" % (char_score_rel[self.GA], char_score_rel[self.EN]))
        #print("%.15f %.15f" % (weighted_score[self.GA], weighted_score[self.EN]))
        return weighted_score
