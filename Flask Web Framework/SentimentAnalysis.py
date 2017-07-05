from flask import Flask, render_template
from wtforms import Form, TextAreaField


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


class ReviewForm(Form):
    reviewText = TextAreaField()


@app.route('/', methods=['GET', 'POST'])
def index():
    reviewform = ReviewForm()
    return render_template("review.html", form=reviewform)


if __name__ == '__main__':
    app.run(debug=True)
