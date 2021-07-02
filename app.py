from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    json_file = {}
    json_file['query'] = 'hello_worldliness'
    print("returning hellow world")
    return jsonify(json_file)
