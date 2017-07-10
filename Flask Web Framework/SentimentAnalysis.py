from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from flask_wtf import FlaskForm


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


class ReviewForm(FlaskForm):
    reviewText = TextAreaField("", validators=[validators.DataRequired(),
                                               validators.length(min=10)])


@app.route('/', methods=['GET', 'POST'])
def index():
    reviewform = ReviewForm()
    if reviewform.validate_on_submit():
        return "<h1>Form submitted successfully !<h1>"
    return render_template("review.html", form=reviewform)


if __name__ == '__main__':
    app.run(debug=True)
