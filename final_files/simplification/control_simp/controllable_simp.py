import argparse

from controllable_simplification.generate_candidates import main as generate_candidates
from controllable_simplification.ranking.main import main as rank
from final_files.simplification.util import groups_of_n
from final_files.util import sentence_tokenizer


def controllable_simp(input, directory="../../../controllable_simplification", n=2):
    print("INPUT")
    print(input)
    text = groups_of_n(n, sentence_tokenizer(input))
    print(text)

    f = open(f"{directory}/DiscourseSimplification/input.txt", "w")
    f.truncate(0)
    f.write(text)
    f.close()

    args = argparse.Namespace(input=f"{directory}/DiscourseSimplification/input.txt",
                              output='candidates.txt')
    generate_candidates(args)

    print("generated")
    args = argparse.Namespace(model=f'{directory}/ranking/model.bin',
                              input=f"{directory}/DiscourseSimplification/input.txt",
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
