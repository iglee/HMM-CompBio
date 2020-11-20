# initial import statements
import pandas as pd
import numpy as np
from numpy import log2

# global variables
NUC_TO_IDX = dict(zip(["A","C","G","T"],range(4)))
IDX_TO_NUC = dict(zip(range(4),["A","C","G","T"]))

def convert_seq_to_idx(seq):
    return [NUC_TO_IDX[x] for x in seq]

class HMM:
    def __init__(self):
        self.seq = None
        self.seq_idx = None
        self.start_proba = [0.9999, 0.0001]
        self.state1_proba = [0.25, 0.25, 0.25, 0.25]
        self.state2_proba = [0.2, 0.3, 0.3, 0.2]
        self.viterbi_trellis = None

    def input_sequence(self, seq):
        self.seq = list(seq.upper())
        self.seq_idx = convert_seq_to_idx(self.seq)
    
    def forward(self):
        return None
    
    def backtrace(self):
        return None