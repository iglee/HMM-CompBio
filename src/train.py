from HMM import read_fna, GenomeData, HMM
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
    for interval in h.intervals[:k]:
        l = h.seq_idx[interval[0]:interval[1]+1]
        new_trans_proba, new_emit_proba = new_proba(l, h.start_proba, h.emit_proba, h.trans_proba)
        h.update_proba(new_emit_proba,new_trans_proba)
    h.viterbi()
    h.backtrace()
    print("\n\n#########################\n Training Iteration {}: \n#########################".format(i+1))
    h.print_report()






def generate_substrings(seq):
    for i in range(1,len(seq)):
        yield seq[i-1:i+1]

def calculate_substring_proba(start_proba, emit_proba, trans_proba, x):
    # calculate probabilities for these pairs
    state_trans_proba = np.array([ trans_proba[0] * start_proba[0] , trans_proba[1] * start_proba[1] ])

    #initial emission proba * final emission proba
    sub_emit_proba = \
        np.array([[emit_proba[0][x[0]]*emit_proba[0][x[1]], emit_proba[0][x[0]]*emit_proba[1][x[1]]],\
                 [emit_proba[1][x[0]]*emit_proba[0][x[1]], emit_proba[1][x[0]]*emit_proba[1][x[1]]]])

    # final probability calculations
    substring_proba = np.multiply(sub_emit_proba, state_trans_proba)

    
    return substring_proba, substring_proba.max(), np.unravel_index(substring_proba.argmax(), substring_proba.shape) 

def normalize_pseudo(pseudo):
    #works for both emission and transmission proba
    return np.array([pseudo[0]/pseudo[0].sum(), pseudo[1]/pseudo[1].sum()])

def generate_pairs(nuc):
    nucleotides = range(4)

    #filter pairs with a nucleotide
    pair_selection = list(filter(lambda n:nuc in n, list(permutations(nucleotides,2)))) + [(nuc,nuc)]
    
    return pair_selection

def possible_emit_probas(start_proba, trans_proba, emit_proba, nucleotide_pair):
    """
    calculate all emission probabilities for given nucleotide
    by considering all possible states
    """
    state_trans_proba = np.array([ trans_proba[0] * start_proba[0] , trans_proba[1] * start_proba[1] ])

    p_obs = []
    for state in possible_states:
        #initial = state[0], final= state[1]
        p_obs.append(emit_proba[state[0]][nucleotide_pair[0]]*emit_proba[state[1]][nucleotide_pair[1]]*state_trans_proba[state[0]][state[1]])

    p_obs = np.array(p_obs)
    p_obs_max = max(p_obs)
    
    return p_obs, p_obs_max

def proba_args(x_idx, state):
    args = []

    for i in range(len(possible_states)):
        if possible_states[i][x_idx] == state:
            args.append(i)
    return args

def pseudo_emit_proba(nucleotide, state, start_proba, trans_proba, emit_proba):
    pairs = generate_pairs(nucleotide)
    max_proba_nuc_state = []
    max_proba_nuc = []

    for pair in pairs:
        p_obs, p_obs_max = possible_emit_probas(start_proba, trans_proba, emit_proba, pair)
        if pair == (nucleotide, nucleotide):
            idx_prob = [possible_states.index((state, state))]
        else:
            idx_nuc = pair.index(nucleotide)
            idx_prob = proba_args(idx_nuc, state)

        max_proba_nuc_state.append(max([p_obs[x] for x in idx_prob]))
        max_proba_nuc.append(p_obs_max) 

    return (sum(max_proba_nuc_state)/sum(max_proba_nuc))

def new_proba(l, start_proba, emit_proba, trans_proba):
    max_proba_total = 0
    pseudo_trans_total = np.zeros((2,2))

    for x in generate_substrings(l):
        pseudo_trans, max_proba, _ = calculate_substring_proba(start_proba, emit_proba, trans_proba, x)
        max_proba_total += max_proba
        pseudo_trans_total += pseudo_trans

    new_trans_proba = normalize_pseudo(pseudo_trans_total/max_proba_total)
    
    pseudo_emit = np.array([[pseudo_emit_proba(x,0, start_proba, trans_proba, emit_proba) for x in range(4)],\
                            [pseudo_emit_proba(x,1, start_proba, trans_proba, emit_proba) for x in range(4)]])
    new_emit_proba = normalize_pseudo(pseudo_emit)
    
    return new_trans_proba, new_emit_proba