import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    #return "Hello World"

#prediction function
def ValuePredictor(to_predict_list, seg):
    #to_predict = np.array(to_predict_list).reshape(1, 12)
    print(to_predict_list)
    loaded_model = pickle.load(open("model_lead_dt.pkl", "rb"))
    # La segmentation Divers influe sur la conversion
    if seg == "7":
        result = loaded_model.predict([[1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0]])
    else:
        result = loaded_model.predict([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                        0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    return result[0]




@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        segmentation = request.form.get('seg')
        print(segmentation)
        to_predict_list = request.form.to_dict()
        print(to_predict_list)
        to_predict_list=list(to_predict_list.values())
        print(to_predict_list)
        #to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list, segmentation)
        
        if int(result)==1:
            prediction='Lead Converted'
        else:
            prediction='Lead Not Converted !'
            
        return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)
