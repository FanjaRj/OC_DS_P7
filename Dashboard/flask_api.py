# Importation de librairies
from flask import Flask,request
import pickle
import numpy as np
import pandas as pd
import sklearn
from zipfile import ZipFile

# Création de l'objet app
app = Flask(__name__)

# Chargement des données
zip_file = ZipFile('data_selected.zip')
data = pd.read_csv(zip_file.open('../Results/data_selected.csv'))
feats = [c for c in data.columns if c not in ['TARGET','SK_ID_CURR']]

# Chargement du modèle de classification
pickle_in = open('../Results/model.pkl','rb')
model = pickle.load(pickle_in)


@app.route("/")
def home():
	return "Flask API Prediction !"

@app.route("/predict")
def predict_target():
    ID = int(request.args.get('ID'))
    try :
        ID_data = data[data['SK_ID_CURR']==ID]
        ID_to_predict = ID_data[feats]
        prediction = model.predict(ID_to_predict)
        proba = model.predict_proba(ID_to_predict)
        if (prediction == 0) | (prediction == 1):
            res =  '{ "target":'+str(int(prediction))+', "risk":%.2f }'%tuple(proba[0])[1]
        else :
            res = "Erreur du programme !"
        return res
    except:
        return "Client introuvable !"


# Chargement de l'objet app
if __name__ == "__main__":
	app.run(debug=True)
