# Copyright 2014 Dimitrios Alikaniotis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Vector-space modelling via Random Indexing
supports both by word and by document
http://www.sics.se/~mange/papers/RI_intro.pdf
'''

__author__ = 'Dimitrios Alikaniotis'
__email__ = 'da352@cam.ac.uk'
__affiliation__ = '\
                University of Cambridge\
                Department of Theoretical and Applied Linguistics'

import os
import sys
import getopt
import random
import logging
import cPickle as pickle

import numpy as np

## np.random.seed(31784)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def usage(): pass

class CorpusReader(object):
    def __init__(self, file): self.file = file
    def __iter__(self):
        with open(self.file) as f:
            try:
                for line in f:
                    yield line.lower().split()
            except IOError:
                logging.error('Please specify the corpus location')

class Model(object):
    def save(self):
        logging.info('Saving dictionary file')
        pickle.dump(self.vocab, open('Random_Indexing_Word_Vectors_{0}_dict.dict'.format(self.dims), "wb"))
        logging.info('Saving word vectors')
        np.save('Random_Indexing_Word_Vectors_{0}'.format(self.dims), self.model)
    
    def load(self): pass
        

class RIModel(Model):
    def __init__(self, corpus=None, dims=1024, non_zeros=6, window=0, ternary=True):
        self.dims = dims
        self.non_zeros = non_zeros
        self.window = window
        self.ternary = ternary
        
        self.ndocs = 0
        self.nwords = 0
        self.vocab = {}


        if corpus is not None:
            self.corpus = CorpusReader(corpus)
            self.buildVocab()
                self.init_vectors()
            if self.window:
                self.train_window()
            else:
                self.train_document()
        
    def buildVocab(self):
        for line in self.corpus:
            self.ndocs += 1
            self.nwords += len(line)
            for word in line:
                if word in self.vocab: continue
                self.vocab[word] = len(self.vocab)
                
        logging.info('\
            Number of documents in the corpus: {0}\n\
            Number of word types: {1}\n\
            Number of word tokens: {2}\n'.format(self.ndocs, len(self.vocab), self.nwords))
        
    def train_document(self):
        for doc_ind, line in enumerate(self.corpus):
            logging.info('Parsing document {0}'.format(doc_ind))
            doc_vec = self.get_random_vector(self.dims)
            for word in line:
                self.model[self.vocab[word], :] += doc_vec
                
    def train_window(self):
        for sentence in self.corpus:
            for pos, word in enumerate(sentence):
                word1_idx = self.vocab[word]
                if word is None: continue
                start = max(0, pos - window)
                for pos2, word2 in enumerate(sentence[start:pos+window+1], start):
                    if word2 and not (pos2 == pos):
                        word2_idx = self.vocab[word2]
                        self.model[word1_idx, :] += self.model[word2_idx, :]
                
    def init_vectors(self, dtype=np.int16):
        self.model = np.zeros((len(self.vocab), self.dims), dtype=dtype)
        if self.window is not None:
            for i in range(len(self.vocab)):
                self.model[i, :] += get_random_vector()
                
            
    def get_random_vector(self):
        '''
        make ternary representation
        initialize vector of zeros and substitute the first
        len(non_zero) elements with evenly distributed
        -1s and 1s then shuffle to get random repr.
        '''
        if self.ternary:
            vector = np.zeros((self.dims), dtype=np.int8)
            ones = [i for i in (-1, 1) for j in range(self.non_zeros / 2)]
            ## if NON_ZEROS odd
            if len(ones) < self.non_zeros:
                ones.append(np.random.choice([-1, 1]))
            vector[:len(ones)] += ones
            np.random.shuffle(vector)
        else:
            vector = np.random.normal(0, np.sqrt(1. / self.dims), size=self.dims)
        return vector
        

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:n:f:o:w:t:", ["file=", "output="])
    except getopt.GetoptError as err:
        sys.exit(str(err))
        usage()
        
    dims = non_zeros = ternary = window = 0
    corpus_file = None
    output = os.getcwd()
    
    for o, a in opts:
        if o == "-d":
            dims = int(a)
        elif o == "-n":
            non_zeros = int(a)
        elif o == "-t":
            ternary = bool(a)
        elif o == "-w":
            window = int(a)
        elif o in ("-f", "--file"):
            if a in os.listdir(os.getcwd()):
                corpus_file = os.path.join(os.getcwd(), a)
            else:
                corpus_file = a
        elif o in ('-o', '--output'):
            output = a
        else:
            assert False, "unhandled option"

    logging.info('Random Indexing model\n-----\n\
    Corpus loaded from: {0}\nNumber of given dimensions: {1}\n\
    Proportion of non-zero elements: {2}\n'.format(corpus_file, dims, non_zeros))
    
    ri = RIModel(corpus=corpus_file, dims=dims, non_zeros=non_zeros,ternary=xiternary,window=window)
    ri.save()


if __name__ == '__main__':
    main(sys.argv[1:])