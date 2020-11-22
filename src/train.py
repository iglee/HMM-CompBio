from HMM import read_fna, GenomeData, HMM
import argparse

# global variable
k = 5

# input arguments
parser = argparse.ArgumentParser(description = "train HMM on an input dataset")
parser.add_argument("--input-file", action="store", help="input dataset")
args = parser.parse_args()

# input file
input_data = read_fna(args.input_file)

# load model
h = HMM()
h.input_sequence(input_data[0].sequence)
h.viterbi()
path = h.backtrace()
print(path)