from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model=joblib.load('model.pkl')
scaler=joblib.load('model.sav')

app =Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def home():
    if request.method =='POST':
        sl=request.form['cons.price.idx']
        sw = request.form['euribor3m']
        pl = request.form['pdays']
        pw = request.form['age']
        data = np.array([[sl, sw, pl, pw]])
        x = scaler.transform(data)
        print(x)
        prediction = model.predict(x)
        if prediction == 1:
            print("El cliente contrata un deposito a largo plazo, feliz y bendecido dia")
        else:
            print("El cliente no quiso contratar el servicio, toca molestar a otro")
    return render_template('index.html',prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)