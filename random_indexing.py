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
    def __init__(self, corpus=None, dims=1024, non_zeros=6):
        self.dims = dims
        self.non_zeros = non_zeros
        self.ndocs = 0
        self.nwords = 0
        self.vocab = {}

        if corpus is not None:
            self.corpus = CorpusReader(corpus)
            self.buildVocab()
            self.train()
        
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
        
    def train(self):
        self.model = np.zeros((len(self.vocab), self.dims), dtype=np.int16)
        for doc_ind, line in enumerate(self.corpus):
            logging.info('Parsing document {0}'.format(doc_ind))
            '''
            make doc ternary representation
            initialize vector of zeros and substitute the first
            len(non_zero) elements with evenly distributed
            -1s and 1s then shuffle to get random repr.
            '''
            doc_vec = np.zeros((self.dims), dtype=np.int8)
            ones = [i for i in (-1, 1) for j in range(self.non_zeros / 2)]
            ## if NON_ZEROS odd
            if len(ones) < self.non_zeros:
                ones.append(np.random.choice([-1, 1]))
            doc_vec[:len(ones)] += ones
            np.random.shuffle(doc_vec)
            for word in line:
                self.model[self.vocab[word], :] += doc_vec

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:n:f:o:", ["file=", "output="])
    except getopt.GetoptError as err:
        sys.exit(str(err))
        usage()
        
    dims = non_zeros = 0
    corpus_file = None
    output = os.getcwd()
    
    for o, a in opts:
        if o == "-d":
            dims = int(a)
        elif o == "-n":
            non_zeros = int(a)
        elif o in ("-f", "--file"):
            corpus_file = os.path.join(os.getcwd(), a)
        elif o in ('-o', '--output'):
            output = a
        else:
            assert False, "unhandled option"

    logging.info('Random Indexing model\n-----\n\
    Corpus loaded from: {0}\nNumber of given dimensions: {1}\n\
    Proportion of non-zero elements: {2}\n'.format(corpus_file, dims, non_zeros))
    
    ri = RIModel(corpus=corpus_file, dims=dims, non_zeros=non_zeros)
    ri.save()


if __name__ == '__main__':
    main(sys.argv[1:])