# initial import statements
import pandas as pd
import numpy as np
from numpy import log

# global variables
NUC_TO_IDX = dict(zip(["A","C","G","T"],range(4)))
IDX_TO_NUC = dict(zip(range(4),["A","C","G","T"]))

def convert_seq_to_idx(seq):
    return [NUC_TO_IDX[x] for x in seq]




# organize input data into a class
class GenomeData:
    def __init__(self):
        self.seq_name = None
        self.sequence = ""
        self.seq_len = 0

def read_fna(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    input_data = []

    for l in lines:
        if l[0] == ">":
            try:
                g.seq_len = len(g.sequence)
                input_data.append(g)
            except:
                pass
            g = GenomeData()
            g.seq_name = l.strip()
        else:
            g.sequence = g.sequence + l.strip().upper()
    
    return input_data




# HMM model
class HMM:
    def __init__(self):
        # sequence data
        self.seq = None
        self.seq_idx = None

        # start probabilities
        self.start_proba = np.array([0.9999, 0.0001])
        self.start_logproba = log(self.start_proba)

        # emission probabilities : emit_proba [state][acgt]
        self.emit_proba = np.array([[0.25, 0.25, 0.25, 0.25],[0.2, 0.3, 0.3, 0.2]])
        self.emit_logproba = log(self.emit_proba)

        # transmission probabilities trans_proba[initial][final]
        self.trans_proba = np.array([[0.9999, 0.0001],[0.01, 0.99]])
        self.trans_logproba = log(self.trans_proba)

        # trellis
        self.viterbi_trellis = None

    def update_proba(self, emit_proba, trans_proba):
        # update emit proba
        self.emit_proba = emit_proba
        self.emit_logproba = log(emit_proba)

        # update transmit proba
        self.trans_proba = trans_proba
        self.trans_logproba = log(trans_proba)


    def input_sequence(self, seq):
        self.seq = np.array(list(seq.upper()))
        self.seq_idx = np.array(convert_seq_to_idx(self.seq))
    
    def forward(self):
        # initialize
        self.viterbi_trellis = np.zeros((2, len(self.seq)))
        self.viterbi_trellis[0][0] = self.start_logproba[0] + self.emit_logproba[0][self.seq_idx[0]]
        self.viterbi_trellis[1][0] = self.start_logproba[1] + self.emit_logproba[1][self.seq_idx[0]]
        
        # iterate and calculate log proba for each cell of trellis
        for i in range(1,len(self.seq_idx)): # i = seq
            for j in range(2): # j = state
                self.viterbi_trellis[j][i] = self.emit_logproba[j][self.seq_idx[i]] + max(self.viterbi_trellis[0][i-1]+self.trans_logproba[0][j], self.viterbi_trellis[1][i-1]+self.trans_logproba[1][j])
                #print(self.trans_logproba[0][j], self.trans_logproba[1][j])

    def backtrace(self):
        # reverse the trellis
        trellis_reversed = self.viterbi_trellis[:,::-1]
        temp = trellis_reversed[:,0].argmax()
        path = [ temp ]

        for i in range(1,trellis_reversed.shape[1]):
            if trellis_reversed[int(not temp),i] > trellis_reversed[temp,i]:
                temp = int(not temp)
            path.append(temp)
        
        return path[::-1]