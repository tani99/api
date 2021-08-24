import os
from multiprocessing.dummy import freeze_support

from final_files.simplification.control_simp.controllable_simp import controllable_simp
from final_files.simplification.muss_simp.muss_simplification import simplify_muss

if __name__ == '__main__':
    freeze_support()

    sent1 = "The Lok Sabha is elected for a term of five years."
    sent2 = "Its life can be extended for one year at a time during a national emergency"

    simp_muss_1 = simplify_muss(sent1, 2)
    simp_muss_2 = simplify_muss(sent2, 2)

    print(os.getcwd())
    control_simp_1 = controllable_simp(sent1, "../controllable_simplification", 2)
    control_simp_2 = controllable_simp(sent2, "../controllable_simplification",  2)

    print("------------ 1 ----------------")
    print(simp_muss_1)
    print(control_simp_1)

    print("------------ 2 ----------------")
    print(simp_muss_2)
    print(control_simp_2)