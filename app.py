import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    glass_features =[x for x in request.form.values()]
    final_features=[np.array(glass_features)]
    prediction=model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text='The Glass Typre Predicted is: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
    
