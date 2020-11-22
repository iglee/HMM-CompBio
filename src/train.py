from HMM import read_fna, GenomeData, HMM
import argparse

N = 10 # total number of iteration for training

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
h.backtrace()
print("#########################\n Training Iteration 1: \n#########################")
h.print_report()

# run viterbi training on the model 
for i in range(1,N):
    print("\n\n#########################\n Training Iteration {}: \n#########################".format(i+1))
    h.print_report()
