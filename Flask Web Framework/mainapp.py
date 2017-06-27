from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/<username>')
def profilepage(username):
    return render_template("profile.html", user=username)


@app.route('/')
def index():
    return "This is the Homepage of Sajeev Rajput"


@app.route('/request', methods=['GET', 'POST'])
def requesttype():
    if request.method == 'POST':
        return 'You are using POST'
    else:
        return 'You are probably using GET method'


if __name__ == "__main__":
    app.run(debug=True)
