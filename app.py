from flask import Flask,request, jsonify,render_template
import pickle
import numpy as np

# load the trained model
model_path = 'trained_model.pkl'
with open(model_path,'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    #make prediction
    prediction = model.predict(final_features)
    output = prediction[0]

    if output == 1:
        return render_template('index.html',prediction_text = 'The Person diabetic 😥')
    else:
         return render_template('index.html',prediction_text = 'The person is Not diabetic😍')
    
if __name__ == "__main__":
    app.run(debug = True)






