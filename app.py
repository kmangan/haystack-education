from flask import Flask, render_template, request
from haystack_setup import haystack_setup

app = Flask(__name__)

pipeline = haystack_setup()  # Set up our Haystack pipeline

def generate_response(question):
    response = pipeline.run({"embedder": {"text": question}, "prompt_builder": {"question": question}})

    answer = response["llm"]["replies"][0]
    print(answer)
    return answer

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        response = generate_response(user_input)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
