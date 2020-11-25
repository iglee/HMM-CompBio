from HMM import read_fna, GenomeData, HMM
from viterbi_train import vt_train
import numpy as np
import argparse
from itertools import permutations


N = 10 # total number of iteration for training
possible_states = [(0,0),(0,1),(1,0),(1,1)] # all possible states with a pair of nucleotides
k = 5

# input arguments
parser = argparse.ArgumentParser(description = "train HMM on an input dataset")
parser.add_argument("-fi", action="store", help="input dataset file")
parser.add_argument("-fo", action="store", help="output file for report")
args = parser.parse_args()

# input file
input_data = read_fna(args.fi)

# load model
h = HMM()
h.input_sequence(input_data[0].sequence)
h.viterbi()
h.backtrace()
print("#########################\n Training Iteration 1: \n#########################")
h.print_report()

# run viterbi training on the model 
for i in range(1,N):
    h = vt_train(h)
    print("\n\n#########################\n Training Iteration {}: \n#########################".format(i+1))
    h.print_report()



