from flask import Flask,render_template, request
import preprocess
import pickle
import numpy as np

model = pickle.load(open('Models/tfidf_cv.pkl','rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
  if request.method == 'POST':
    q1 = request.form['inp1']
    q2 = request.form['inp2']
    query = preprocess.preprocessing(q1,q2)
    prediction = model.predict(query)
    prediction_prob_score = model.predict_proba(query)
    print(prediction_prob_score)
    print(prediction_prob_score[0])
    pos = np.argmax(prediction_prob_score)
    score = prediction_prob_score[0][pos]
    # prediction_score = prediction_prob_score[np.argmax(prediction_prob_score)]
    # print(prediction_score)

    return render_template('index.html', prediction=prediction[0], prediction_score1=score)

if __name__ == '__main__':
    app.run(debug=True)
