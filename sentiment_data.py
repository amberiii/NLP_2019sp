# sentiment_data.py

from typing import List
from utils import *
import re
import numpy as np


# Wraps a sequence of word indices with a 0-1 label (0 = negative, 1 = positive).
# Consider augmenting these instances with cached features if feature extraction proves
# to be a bottleneck.
class SentimentExample:
    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()


# Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences.
def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    f = open(infile, encoding='iso8859')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            # Slightly more robust to reading bad output than int(fields[0])
            label = 0 if "0" in fields[0] else 1
            sent = fields[1]
            tokenized_cleaned_sent = list(filter(lambda x: x != '', _clean_str(sent).rstrip().split(" ")))
            # tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            exs.append(SentimentExample(tokenized_cleaned_sent, label))
    f.close()
    return exs


# Writes sentiment examples to an output file in the same format they are read in. Note that what gets written
# out is tokenized, so this will not exactly match the input file. However, this is fine from the standpoint of writing
# model output.
def write_sentiment_examples(exs: List[SentimentExample], outfile: str):
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]) + "\n")
    o.close()


# Tokenizes and cleans a string: contractions are broken off from their base words, punctuation is broken out
# into its own token, junk characters are removed, etc. For this corpus, punctuation is already tokenized, so this
# mainly serves to handle contractions (it's) and break up hyphenated words (crime-land => crime - land)
def _clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\-", " - ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return string
