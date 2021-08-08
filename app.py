from final_files.summarisation.edmundsons.run_edmundsons import Edmundsons
from final_files.util import sentence_tokenizer
from simplification.summariser import summarise_text
from flask import Flask, jsonify, render_template

from tables.tables import k_means_cluster

app = Flask(__name__, template_folder='templates')

DEFAULT_SUMMARY = "default summary"

@app.route("/")
def index():
    return render_template('summarise.html', summary=DEFAULT_SUMMARY)

@app.route('/hello')
def hello():
    json_file = {}
    json_file['query'] = 'lello lello'
    return jsonify(json_file)

@app.route('/world')
def world():
    json_file = {}
    json_file['query'] = 'worldy'
    return jsonify(json_file)

# text = "The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13th Lok Sabha, Bhartiya Janata Party lost a no-confidence motion by one vote and had to resign."
# @app.route('/summarise/', methods=['GET','POST'])
# async def summarise():
#     summary = "Something went wrong"
#     if request.method == "POST":
#         text = request.form.get("text")
#         summary = await summarise_text(text)

#     return render_template('summarise.html', summary=summary)

@app.route('/summarise/<text>', methods=['GET', 'POST'])
def summarise(text):
    # ALL
    sentences_count = int(0.8*len(sentence_tokenizer(text)))
    summariser = Edmundsons(text, sentences_count, cue_weight=1.0, key_weight=1.0,
                            title_weight=1.0, length_weight=1.0)
    summary = summariser.summarise()

    print("got text: ", text)
    json_file = {}
    json_file['summary'] = summary
        # getSummary(text)
    print(json_file)
    return jsonify(json_file)



def getSummary(text):
    return summarise_text(text, simplifier="muss")

def getSummaryTest(text):
    return "TEST: " + text


@app.route('/tabulate/<text1>/<text2>', methods=['GET', 'POST'])
def tabulate(text1, text2):
    print("got text: ", text1, text2)
    tabulated = k_means_cluster(text1, text2)
    # json = tabulated.to_json()
    return tabulated
    # print(json)
    # json_file = {}
    # json_file['summary'] = getSummary(text)
    # print(json_file)
    # return jsonify(json_file)