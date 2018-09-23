from flask import Flask, session, redirect, url_for, escape, request, render_template
from experiment import Experiment

global app, exp

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['query'] = request.form['query']
        return redirect(url_for('search'))
    else:
        return render_template('index.html')

@app.route('/s', methods=['GET', 'POST'])
def search():
    emotion = exp.predict(session['query'])
    if emotion:
        result = '<br /> I think the emotion is: <b>{}</b>.<br />'.format(emotion)
    else:
        result = 'Sorry, I don\'t know what emotion is there <br />'
    return render_template('results.html') + result  + render_template('tail.html')

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


if __name__ == "__main__":
    global exp
    exp = Experiment('test demo')
    app.run(host='0.0.0.0',port=4200)
