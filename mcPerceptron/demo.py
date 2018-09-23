from flask import Flask, session, redirect, url_for, escape, request, render_template
from experiment import Experiment

global app, exp, query
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
                '<br />Tweet text was:<br /><blockquote class="twitter-tweet"' \
                ' data-partner="tweetdeck"><p dir="ltr">"{}"</p></blockquote>' \
                '</center>'.format(emotion, query)
        else:
            result = '<center><br />Sorry, I don\'t know what emotion is ' \
                'there <br /></center>'
        del query
        return render_template('index.html') + result
    else:
        intro = '<center><br />Hi! I\'m a computer program who tries detects ' \
            'imlicit emotions in Tweets.<br /> For now I understand only ' \
            'English. <br /><br /></center>'
        return render_template('index.html') + intro

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


if __name__ == "__main__":
    global exp
    exp = Experiment('test demo')
    app.run(host='0.0.0.0',port=4200)
