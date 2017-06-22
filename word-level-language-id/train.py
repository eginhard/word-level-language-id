# -*- coding: utf-8 -*-

from LanguageModel import LanguageModel

ngram = 2

# Language codes
fr = 'ga'
en = 'en'

# Language model files.
fr_lm_fn = 'word-level-language-id/models/ga.lm'
en_lm_fn = 'word-level-language-id/models/en.lm'

# Unigram frequency lexicons.
fr_lex = 'word-level-language-id/corpora/ga-words.txt'
en_lex = 'word-level-language-id/corpora/en-GB-words.txt'

# Load lexicons.
fr_lm = LanguageModel(fr, ngram, fr_lex)
en_lm = LanguageModel(en, ngram, en_lex)

# Train and save character n-gram models.
fr_lm.train()
fr_lm.dump(fr_lm_fn)

en_lm.train()
en_lm.dump(en_lm_fn)
