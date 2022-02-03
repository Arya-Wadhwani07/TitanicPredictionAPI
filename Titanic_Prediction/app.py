import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
pickle_in = open(r'D:/Work Stuff/Python Programs/Model.pckl', 'rb')
pickle_model = pickle.load(pickle_in)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    request_data = request.get_json()
    PassengerId= request_data['PassengerId']
    Pclass=request_data['Pclass']
    Age= request_data['Age']
    SibSp= request_data['SibSp']
    Parch= request_data['Parch']
    Fare= request_data['Fare']
    Sex_female= request_data['Sex_female']
    Sex_male= request_data['Sex_male']
    Embarked_C= request_data['Embarked_C']
    Embarked_Q= request_data['Embarked_Q']
    Embarked_S= request_data['Embarked_S']
    data = [PassengerId,Pclass,Age,SibSp,Parch,Fare,Sex_female,Sex_male,Embarked_C,Embarked_Q, Embarked_S]
    data = np.array(data)
    pred = pickle_model.predict(data.reshape(1,-1))
    if pred==0:
        return "Not Survived"
    return "Survived"

if __name__ == '__main':
    app.run(debug=True)