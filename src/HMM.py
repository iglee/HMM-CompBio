# initial import statements
import pandas as pd
import numpy as np
from numpy import log

# global variables
# global variables
NUC_TO_IDX = dict(zip(["A","C","G","T"],range(4)))
IDX_TO_NUC = dict(zip(range(4),["A","C","G","T"]))

def convert_seq_to_idx(seq):
    return [NUC_TO_IDX[x] for x in seq]

def read_fna(filename):
    return None

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

    def input_sequence(self, seq):
        self.seq = np.array(list(seq.upper()))
        self.seq_idx = np.array(convert_seq_to_idx(self.seq))
    
    def forward(self):
        # initialize
        self.viterbi_trellis = np.zeros((2, len(self.seq)))
        self.viterbi_trellis[0][0] = self.start_logproba[0] + self.emit_logproba[0][h.seq_idx[0]]
        self.viterbi_trellis[1][0] = self.start_logproba[1] + self.emit_logproba[1][h.seq_idx[0]]
        
        # iterate and calculate log proba for each cell of trellis
        for i in range(1,len(h.seq_idx)): # i = seq
            for j in range(2): # j = state
                h.viterbi_trellis[j][i] = h.emit_logproba[j][h.seq_idx[i]] + max(h.viterbi_trellis[j][i-1]+h.trans_logproba[0][j], h.viterbi_trellis[j][i-1]+h.trans_logproba[1][j])


    def backtrace(self):
        return None