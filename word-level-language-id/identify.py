# -*- coding: utf-8 -*-

from LanguageIdentifier import LanguageIdentifier

# Language codes
fr = 'ga'
en = 'en'

# Language model files.
fr_lm = 'word-level-language-id/models/ga.lm'
en_lm = 'word-level-language-id/models/en.lm'

# Unigram frequency lexicons.
fr_lex = 'word-level-language-id/corpora/ga-words.txt'
en_lex = 'word-level-language-id/corpora/en-GB-words.txt'

identifier = LanguageIdentifier(fr, en,
                                fr_lm, en_lm,
                                fr_lex, en_lex)

# Tokenized input strings.
sentences = ["The name of the State is Éire or in the English language Ireland",
             "Dáil Éireann shall be summoned and dissolved by the President on the advice of the Taoiseach",
             "táim ag dul ar tinder date"]
sentences = map(lambda x: x.split(), sentences)

for tokens in sentences:
    print(" ".join(tokens))
    print(list(zip(tokens, identifier.identify(tokens))))
    print(" ")
