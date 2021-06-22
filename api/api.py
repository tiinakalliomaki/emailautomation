import flask
from flask import request, jsonify
import sys
import os
sys.path.insert(0, '../ml_ds')
import encode_score2

app = flask.Flask(__name__)
app.config["DEBUG"] = True
@app.route('/', methods=['POST'])
def home():
    body_text=request.headers.get('email-body-text')
    thanks_similarity=encode_score2.get_score(body_text)
    if thanks_similarity<.4:
        return jsonify(1)
    else:
        return jsonify(0)


if __name__ == "__main__":
    app.debug = True
    app.run()