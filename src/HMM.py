# initial import statements
import pandas as pd
import numpy as np
from numpy import log
import re
import itertools
from tabulate import tabulate

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
            g.sequence = g.sequence + re.sub(r"((?:(?!A|C|T|G)\S))","T",l.strip().upper())

    g.seq_len = len(g.sequence)
    input_data.append(g)

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
        self.states = None

        # optimal path
        self.path = None
        self.hits = None
        self.intervals = None
        self.k = 5 # number of hits to consider for training

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
    
    def viterbi(self):
        # initialize
        self.viterbi_trellis = np.zeros((2, len(self.seq)))
        self.states = np.zeros((2, len(self.seq))) # track prev states in each cell of the array. with first prev state = 0 (will not be used)
        
        self.viterbi_trellis[0][0] = self.start_logproba[0] + self.emit_logproba[0][self.seq_idx[0]]
        self.viterbi_trellis[1][0] = self.start_logproba[1] + self.emit_logproba[1][self.seq_idx[0]]
        
        # iterate and calculate log proba for each cell of trellis
        for i in range(1,len(self.seq_idx)): # i = seq
            for j in range(2): # j = state
                
                temp_score = self.viterbi_trellis[:,i-1]+self.trans_logproba[:,j]
                
                self.states[j][i] = temp_score.argmax()
                self.viterbi_trellis[j][i] = self.emit_logproba[j][self.seq_idx[i]] + max(temp_score)


    def backtrace(self):
        last_max = self.viterbi_trellis[:,-1].argmax()
        path = [ last_max ]

        for i in range(self.states.shape[1]-1, 0, -1):
            last_max = int(self.states[last_max][i])
            path.append(last_max)

        self.path = np.array(path[::-1])
        self.hits = np.where(self.path == 1)[0] # record the index of states
        self.find_hit_sequences(self.hits)

    def find_hit_sequences(self, hit_locations):
        hit_intervals = []
        for _, group in itertools.groupby(enumerate(hit_locations), key=lambda x: x[1]-x[0]):
            group = list(group)
            hit_intervals.append((group[0][1],group[-1][1]))
        self.intervals = hit_intervals

    def print_report(self, file):
        # print emission probabilities
        emit_table = [list(self.emit_proba[0]), list(self.emit_proba[1])]
        emit_table[0].insert(0, "State 1")
        emit_table[1].insert(0, "State 2")
        print("\nEmission probabilities:\n", file=file)
        print(tabulate(emit_table, headers=['A', 'C', 'G', 'T']), file=file)
        print("\nEmission probabilities:\n")
        print(tabulate(emit_table, headers=['A', 'C', 'G', 'T']))

        # print transmission probabilities
        transmit_table = [list(self.trans_proba[0]), list(self.trans_proba[1])]
        transmit_table[0].insert(0, "State 1")
        transmit_table[1].insert(0, "State 2")
        print("\n\nTransmission probabilities:\n", file=file)
        print(tabulate(transmit_table, headers=["State 1", "State 2"]), file=file)
        print("\n\nTransmission probabilities:\n")
        print(tabulate(transmit_table, headers=["State 1", "State 2"]))

        # print log probability of viterbi path
        logproba_path = self.viterbi_trellis[:,-1].max()
        print("\n\nLog probability of the viterbi path: ", logproba_path, file=file)
        print("\n\nLog probability of the viterbi path: ", logproba_path)

        # print total number of hits
        print("\n\nTotal number of hits: ", len(self.intervals), file=file)
        print("\n\nTotal number of hits: ", len(self.intervals))

        # print length and locations of first k hits
        k_intervals = self.intervals[:self.k]
        print("\n\nLengths and locations of first {} hits".format(self.k), file=file)
        print("\n\nLengths and locations of first {} hits".format(self.k))
        for x in k_intervals:
            print("Interval location: ", x, file=file)
            print("Interval length: ", x[1]-x[0], file=file)
            print("\n", file=file)

            print("Interval location: ", x)
            print("Interval length: ", x[1]-x[0])
            print("\n")

        print("\n\n\n", file=file)
        print("\n\n\n")

