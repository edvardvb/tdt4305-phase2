import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-training', help='path to training set', type=str)
parser.add_argument('-input', help='path to input file', type=str)
parser.add_argument('-output', help='path to output file', type=str)
args = parser.parse_args()

print(args.training)
print(args.input)
print(args.output)
