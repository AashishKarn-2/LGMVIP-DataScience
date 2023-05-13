from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods= ['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST' ,'GET'])
def predict():
    import pickle
    import numpy as np

    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get the input values from the form
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])

    # Make a prediction
    input_data = np.array([[petal_length, petal_width, sepal_length, sepal_width]])

    # Make a prediction using the model
    prediction = model.predict(input_data)[0]
    flower = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    # Get the corresponding flower name from the flower array
    flower_name = flower[int(prediction)]

    return render_template('predict.html', prediction=flower_name)


if __name__ == '__main__':
    app.run(debug=True)
