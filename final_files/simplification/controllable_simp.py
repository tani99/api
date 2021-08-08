
import argparse

from controllable_simplification.generate_candidates import main as generate_candidates
from controllable_simplification.ranking.main import main as rank
from final_files.util import sentence_tokenizer


def controllable_simp(input):
    text = " ".join(sentence_tokenizer(input))
    f = open("../../controllable_simplification/DiscourseSimplification/input.txt", "w")
    f.truncate(0)
    f.write(text)
    f.close()

    args = argparse.Namespace(input="../../controllable_simplification/DiscourseSimplification/input.txt", output='candidates.txt')
    generate_candidates(args)

    print("generated")
    args = argparse.Namespace(model='../../controllable_simplification/ranking/model.bin', input="../../controllable_simplification/DiscourseSimplification/input.txt",
                              candidates='candidates.txt', output='output.txt')
    rank(args)
    print("simplified")
    simplified = ""
    with open('output.txt') as f:
        print("printing contents")
        contents = f.read()
        simplified += contents
        print(contents)

    return simplified