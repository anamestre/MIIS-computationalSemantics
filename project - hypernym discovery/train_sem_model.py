import os
import sys

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class AdaptedLineSentence(object):
    """
    Adapted source code from LineSentence (in gensim.models.word2vec).
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, source, lowercase=False):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.lowercase = lowercase is True
        self.lines_read = 0
        
    def preprocess(self, line):
        if self.lowercase is True:
            return line.lower()
        return line

    def __iter__(self):
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                yield self.preprocess(utils.to_unicode(line)).split()
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with gensim.utils.smart_open(self.source) as fin:
                for line in fin:
                    yield self.preprocess(gensim.utils.to_unicode(line)).split()
                    self.lines_read += 1

#corpus_fname = "1A_en_UMBC_tokenized.tar.gz"
corpus_fname = "1C_es_1Billion_tokenized.tar.gz" 
my_reader = AdaptedLineSentence(corpus_fname, lowercase=True)
model = Word2Vec(my_reader, sg=1, size=300, workers=1) 
model.save("spanish.model")  # save the model