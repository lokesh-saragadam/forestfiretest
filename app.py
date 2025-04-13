import pickle
from flask import Flask,request,jsonify,render_template
import pandas as ps
import numpy as np
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app =application

scaler = StandardScaler()
##import ridge and scaler model

ridge_model = pickle.load(open('models/model.pkl','rb'))
scaler_model = pickle.load(open('models/scaling.pkl','rb'))
@app.route("/index",methods =['GET','POST'])
def index():
  return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            temperature = float(request.form.get("temperature"))
            rh = float(request.form.get("rh"))
            ws = float(request.form.get("ws"))
            rain = float(request.form.get("rain"))
            ffmc = float(request.form.get("ffmc"))
            dmc = float(request.form.get("dmc"))
            isi = float(request.form.get("isi"))
            region = int(request.form.get("region"))
            classes = int(request.form.get("class"))  # optional

            # Prepare and scale input
            input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, region, classes]])
            scaled_input = scaler_model.transform(input_data)
            result = ridge_model.predict(scaled_input)[0]

            resu = f"The FWI predicted is {result:.2f}"
            return render_template('home.html', results=resu)
        
        except Exception as e:
            return render_template('home.html', results=f"Error occurred: {str(e)}")

    return render_template('home.html', results=None)


if __name__ == "__main__":
  app.run(host = "0.0.0.0")