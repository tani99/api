from simplification.edmundson import edmundsons
from simplification.muss_simp import simplify_muss
from simplification.control_simp import controllable_simplification

# from edmundson import edmundsons
# from muss_simp import simplify_muss
# from control_simp import controllable_simplification

from multiprocessing import Process, freeze_support

def summarise_text(text, simplifier="muss", percentage=0.8):
    extracted = edmundsons(text, percentage)
    summary = ""
    if simplifier == "muss":
        # MUSS
        summary = simplify_muss(extracted)
    elif simplifier == "control-simp":
        # Controlled simplification
        summary = controllable_simplification(extracted)

    return summary

if __name__ == '__main__':
    freeze_support()
    text = "The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13th Lok Sabha, Bhartiya Janata Party lost a no-confidence motion by one vote and had to resign."
    print(summarise_text(text))
    