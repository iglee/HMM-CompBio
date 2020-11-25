from HMM import read_fna, GenomeData, HMM
from viterbi_train import vt_train
from test import match
import numpy as np
import argparse
import pandas as pd


N = 10 # total number of iteration for training
possible_states = [(0,0),(0,1),(1,0),(1,1)] # all possible states with a pair of nucleotides
k = 5

# input arguments
parser = argparse.ArgumentParser(description = "train HMM on an input dataset")
parser.add_argument("-fi", action="store", help="input dataset file")
parser.add_argument("-fo", action="store", help="output file for report")
parser.add_argument("-gt", action="store", help="golden test set")
args = parser.parse_args()

# input file
input_data = read_fna(args.fi)

# output file
out_file = open(args.fo, "w")

# load model
h = HMM()
h.input_sequence(input_data[0].sequence)
h.viterbi()
h.backtrace()
print("#########################\n Training Iteration 1: \n#########################", file=out_file)
print("#########################\n Training Iteration 1: \n#########################")
h.print_report(out_file)

# run viterbi training on the model 
for i in range(1,N):
    h = vt_train(h)
    print("\n\n#########################\n Training Iteration {}: \n#########################".format(i+1),file=out_file)
    print("\n\n#########################\n Training Iteration {}: \n#########################".format(i+1))
    h.print_report(out_file)


# evaluation on the final model
df=pd.read_csv(args.gt, sep="\t", skiprows=9, header = None) # golden dataset
print("Evaluating against a golden dataset...\n\n")
print("Evaluating against a golden dataset...\n\n", file = out_file)
for x in h.intervals:
    print("matches for intervals {}".format(x))
    print("matches for intervals {}".format(x), file = out_file)
    for row in df[1:].iterrows():
        
        try:
            y = (int(row[1][3])-1, int(row[1][4])-1)
            if match(x,y):
                
                print(row[0], y, row[1])
                print(row[0], y, row[1][8], file = out_file)
        except:
            #print("can't match, bad input",row[0])
            pass
            
    print("###\n")
    print("###\n", file = out_file)
    
    
out_file.close()

