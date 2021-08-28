import spacy
from nltk import word_tokenize

from final_files.extraction.extraction import process_text
from final_files.simplification.muss_simp.muss_simplification import simplify_muss
from final_files.summarisation.edmundsons.run_edmundsons import Edmundsons
from final_files.util import sentence_tokenizer
from final_files.visualisation.embeddings import spacy_embedding
from final_files.visualisation.timeline import identify_dates
from final_files.visualisation.weakest_link_method import tabulate_text, get_clusters
from simplification.summariser import summarise_text
from flask import Flask, jsonify, render_template

from tables.tables import k_means_cluster
from util import get_keywords

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

@app.route('/keywords/<text>', methods=['GET', 'POST'])
def keywords(text):
    keywords = get_keywords(text)

    json_file = {}
    json_file['keywords'] = keywords
    return jsonify(json_file)


nlp = spacy.load("en_core_web_sm")
# Merge noun phrases and entities for easier analysis
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

@app.route('/summarise/<text>/<percent>/<sum>/<simp>/<points>/<include_first_half>', methods=['GET', 'POST'])
def summarise(text, percent, sum, simp, points, include_first_half):
    # ALL
    temp = text
    if sum == 'true':
        sentences_count = max(1, int((float(percent) / 100) * len(sentence_tokenizer(temp))))
        summariser = Edmundsons(temp, sentences_count, cue_weight=1.0, key_weight=1.0,
                                title_weight=1.0, length_weight=1.0)
        temp = summariser.summarise()

    if simp == 'true':
        print(temp)
        print(type(temp))
        temp = simplify_muss(temp, 2)
        print("Siplified")

    if points == 'true':
        if include_first_half == 'true':
            print('its true!')
            temp = process_text(temp, True)
        else:
            print("Its false!")
            temp = process_text(temp, False)

        print("points")
        print(temp)

    final = temp

    keywords = get_keywords(final)

    doc = nlp(final)
    print("FINAL: ", final)
    json_file = {}
    json_file['summary'] = [token.text for token in doc]
    json_file['keywords'] = keywords
    json_file['summary_original'] = final
    # getSummary(text)
    print(json_file)
    return jsonify(json_file)


def getSummary(text):
    return summarise_text(text, simplifier="muss")


def getSummaryTest(text):
    return "TEST: " + text


@app.route('/tokenize/<text>', methods=['GET', 'POST'])
def words(text):
    doc = nlp(text)
    json_file = {}
    json_file['words'] = [token.text for token in doc]

    return jsonify(json_file)
    # return get_keywords(text)


@app.route('/timeline/<text>/<percent>/<sum>/<simp>/<points>/<include_first_half>/<n>', methods=['GET', 'POST'])
def timeline(text, n, percent, sum, simp, points, include_first_half):
    print("got text: ", text)
    summarised = summarise(text, percent, sum, simp, points, include_first_half).get_json()['summary_original']

    clusters = get_clusters(summarised, int(n), spacy_embedding)
    json = identify_dates(clusters)

    print("DONE!")
    return json


@app.route('/tabulate/<text1>/<text2>/<percent>/<sum>/<simp>/<points>/<include_first_half>/<n>',
           methods=['GET', 'POST'])
def tabulate(text1, text2, percent, sum, simp, points, include_first_half, n):
    print("got text: ", text1, text2)
    summarised1 = summarise(text1, percent, sum, simp, points, include_first_half).get_json()['summary_original']
    summarised2 = summarise(text2, percent, sum, simp, points, include_first_half).get_json()['summary_original']
    print("summarised")
    # tabulated = k_means_cluster(text1, text2)
    s1, s2, table = tabulate_text(summarised1, summarised2, int(n), spacy_embedding)

    return table

    # print("table not issue")
    # json = tabulated.to_json()
    # print("DONE!")
    # print(table.transpose().to_json())
    # return table.transpose().to_json()
    # print(json)
    # json_file = {}
    # json_file['summary'] = getSummary(text)
    # print(json_file)
    # return jsonify(json_file)
