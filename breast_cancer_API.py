from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

loaded_model=pickle.load(open('Random_Forest.pkl','rb'))

cols=['texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se']

@app.route("/")
def getModel():
 return render_template("form.html")
 #return str(loaded_model)

@app.route("/predict",methods=["POST"])
def predict():
 #Load the data
 data=pd.read_csv('data.csv')


 #Get all the predictors
 X=data.loc[:,cols]

 input_data=[]

 for col in cols:
  input_data.append(float(request.form[col]))


 #Normalize the data
 df_norm=pd.DataFrame((input_data-X.mean())/(X.max()-X.min())).transpose()


 #Prediction
 pred=loaded_model.predict(df_norm)

 if pred==1:
     return 'Prediction : Benign Tumor Found'
 else:
     return 'Prediction : Malignant Tumor Found'

if __name__=='__main__':
 app.run(debug=True)
 #app.run(host='0.0.0.0')

 #By default the HOST is 127.0.0.1 that is local host but the problem with
 #using that is if i start running the container with local host then i will not be
 #able to access the local host of the container from the local host of the host.
 #They both are seperate entities untill a bridge is built.

#To connect the front end and the API:
#Create a folder called templates and place form.html inside it.
