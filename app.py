from flask import Flask, render_template, request
import pandas as pd
import pickle

model = pickle.load(open("RF_model_example.pkl", "rb" ))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":
        #load the model in memory

        form_input = pd.DataFrame(request.form.to_dict(), index = [0])
        prediction = model.predict(form_input.astype(float))
        return str(prediction)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)