from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>This is the Homepage</h1>'


@app.route('/shopping')
def shopping():
    shoplist = ['Brush', 'Biscuits', 'Shower Gel']
    return render_template("shopping.html", shoplist=shoplist)


if __name__ == '__main__':
    app.run(debug=True)