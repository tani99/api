import os
import argparse

from controllable_simplification import dissim

import sys
sys.path.insert(0, '../controllable_simplification/ranking')

discourse_simplification_directory = '../../../controllable_simplification/DiscourseSimplification'
def main(args):
    cwd = os.getcwd()
    print(cwd)
    # Runs DisSim to generate candidates
    print("POINT 1")
    os.system("cp " + args.input + f" {discourse_simplification_directory}/input.txt")
    print("POINT 2")
    os.chdir(f'{discourse_simplification_directory}')
    print("POINT 3")
    os.system("mvn clean compile exec:java")
    print("POINT 4")
    os.chdir(cwd)
    print("THIS POINT")
    print("input testing ", args.input)

    dissim_candidates = dissim.generate_candidates(args.input, f"{discourse_simplification_directory}/output_dt.txt")

    # TODO: add neural splitter candidates.

    fpout = open(args.output, "w")
    for candidates in dissim_candidates:
        fpout.write("\t".join(candidates) + "\n")
    fpout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate DisSim candidates that have undergone splitting and deletion.')
    parser.add_argument('--input', help="Input sentences with one sentence in each line.")
    parser.add_argument('--output', help="Candidates for each input sentence seperated by tabs. \n"
                                         "The format for each candidate is "
                                         "<candidate>|||<DisSim|Transformer>|||<Rules applied to obtain the candidate>")
    args = parser.parse_args()
    main(args)
