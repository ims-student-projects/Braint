
import logging
from flask import Flask, session, redirect, url_for, escape, request, render_template
from experiment import Experiment

global app, exp, query, logger

logger = logging.getLogger('log_demo')
_h = logging.FileHandler('log_demo')
_h.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
logger.addHandler(_h)
logger.setLevel(logging.INFO)

query = None
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    global query
    if request.method == 'POST':
        query = request.form['query']
        return redirect(url_for('home'))
    elif query:
        emotion = exp.predict(query)
        if emotion:
            result = '<center><br /> I think the emotion is: <b>{}</b>.<br />' \
                '<br />Tweet text was:<p dir="ltr"><i>{}</i></p>' \
                '</center>'.format(emotion, query)
            logger.info(' ({}) {}?\t{}'.format(request.remote_addr, emotion, query))
        else:
            result = '<center><br />Sorry, I don\'t know what emotion is ' \
                'there <br /></center>'
        query = None
        return render_template('index.html') + result
    else:
        intro = '<center><br />Hi! I\'m a computer program who tries to detect ' \
            'implicit emotions in Tweets.<br /> For now I understand only ' \
            'English. <br /><br /></center>'
        return render_template('index.html') + intro

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


if __name__ == "__main__":
    global exp
    exp = Experiment('test demo')
    app.run(host='0.0.0.0',port=2222)
