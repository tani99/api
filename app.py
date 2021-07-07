from simplification.summariser import summarise_text
from flask import Flask, jsonify, render_template, request, redirect
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
import asyncio

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
    print("got text: ", text)
    json_file = {}
    json_file['summary'] = getSummary(text)
    print(json_file)
    return jsonify(json_file)

def getSummary(text):
    return summarise_text(text)