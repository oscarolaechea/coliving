#Importaciones
import flask
from flask import jsonify
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv
from surprise import SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
#Importando funciones trabajadas de ML:

from Defiant_Recommender_notebook import preprocess_form
from Defiant_Recommender_notebook import defiant_recommender
from Defiant_Recommender_notebook import check_activities_user


#Importamos Modelo:

def load_model():
    with open(r'C:\Users\oscar\OneDrive\Escritorio\Oscar\Curso The Bridge Data Science\thebridge_ft_nov21\Desafio Tripulaciones\DF_APP\model\DEFIANT_RECOMMENDER.model', "rb") as archivo_entrada:
        defiant_pickle = pickle.load(archivo_entrada)
    return defiant_pickle

#Importamos Datos:
def load_data():
    df_cohousing = pd.read_csv(r"C:\Users\oscar\OneDrive\Escritorio\Oscar\Curso The Bridge Data Science\thebridge_ft_nov21\Desafio Tripulaciones\DF_APP\model\data\cohousing_TEMP.csv")
    return df_cohousing

#Transformamos Data:

df= preprocess_form(load_data())


app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Ponemos el título del home:

@app.route('/')



def home():
    return "<h1>MODELO RECOMENDADOR</h1><p>modelo recomendador de actividades a realizar en un cohousing</p>"


# Devolvemos en un json los resultados del modelo recomendador:


@app.route('/api/model/recomendacion', methods=['GET'])
def recomendacion():
    predicion1=defiant_recommender(userId=3,dataframe=df,algorithm=load_model(),n_recommendations=5,column_iid='itemId',column_uid='userId')
    return jsonify(predicion1)

#Devolvemos en un json el top 5 de actividades mejor puntuadas por el usuario X:

@app.route('/api/model/actividades', methods=['GET'])
def actividades():

  
    actividades_1=check_activities_user(dataframe=df,userId=3,n=5,column_rating='rating',column_uid='userId')
    lista_actividades=list(actividades_1.loc[:,'itemId'])
    return jsonify(lista_actividades)

if __name__ == '__main__':
    app.run()
