# Word-level language ID
Simple word-level language identification using the Viterbi algorithm based on unigram frequencies and character n-grams.

## Usage

I recommend using Python 3 for better Unicode support.

To quickly try out the system, corpora and language models are already included for British English and Irish. See below how to add new ones. You might want to do some post-processing on the lexicons because e.g. the Irish one contains some English as well and vice versa.

Run word-level language ID on some example sentences:

```bash
python word-level-language-id/identify.py
```

## Train new language models

Create or download a unigram frequency lexicon, e.g. from the [Crúbadán Project](http://crubadan.org/) which has those readily available for over 2000 languages.

For example, download and unzip British English and Irish:

```bash
wget http://crubadan.org/files/en-GB.zip 
wget http://crubadan.org/files/ga.zip

unzip '*.zip' -d word-level-language-id/corpora
```

Train the language models.

```bash
python word-level-language-id/train.py
```