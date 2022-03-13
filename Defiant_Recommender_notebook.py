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

# Loading DATA

df_cohousing = pd.read_csv(r"C:\Users\oscar\OneDrive\Escritorio\Oscar\Curso The Bridge Data Science\thebridge_ft_nov21\Desafio Tripulaciones\DF_APP\model\data\cohousing_TEMP.csv")
df_cohousing.head()

def preprocess_form(dataframe):
    """
    This functions will preprocess the original csv from the Google form and get it ready for the model.

    Parameters
    -----------

    dataframe (object): the DataFrame containing three columns; userID, itemID and rating.

    return
    ------

    A new dataframe ready for the model.

    """
    #we clean the df from only nan answers and we erase the first two columns as they are not answers for the algorithm
    #dataframe = dataframe[dataframe.columns[2:]]
    dataframe.drop(['Timestamp', '¿Qué edad tienes?'], axis=1, inplace=True)
    dataframe.dropna(how='all', inplace= True)
    
    #we need to change the answers from the form from str to int
    dicc_formulario = {
        '1 No me gusta': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5 Me encanta': 5
    }

    for i in dataframe.columns:
        dataframe[i] = dataframe[i].map(dicc_formulario)

    #we transorm the dataframe to get the triplet column format we need to feed our algorithm
    df_new = pd.DataFrame([(1,1,1)])
    for i in dataframe.index:
        for j in dataframe.columns:
            if dataframe[j][i] == 'NaN':
                pass
            else:
                df_new = df_new.append([(i,j, dataframe[j][i])])

    df_new.dropna(how= 'any', inplace=True)
    df_new = df_new[1:]

    #we cahange the columns names
    df_new.columns = ['userId', 'itemId', 'rating']

    #we create a dictionary to map the name of activities and change it for their activity code
    dict_activities = {
    'Lista de actividades: [YOGA]': 1, 
    'Lista de actividades: [NATACIÓN ]': 2,
    'Lista de actividades: [BAILE ]': 3, 
    'Lista de actividades: [GOLF ]': 4,
    'Lista de actividades: [GIMNASIO ]': 5,
    'Lista de actividades: [TIRO CON ARCO ]': 6,
    'Lista de actividades: [ZUMBA ]': 7, 
    'Lista de actividades: [TENIS]': 8,
    'Lista de actividades: [CLUB DE LECTURA]': 9,
    'Lista de actividades: [CLUB DE ESCRITURA]': 10,
    'Lista de actividades: [PINTURA]': 11, 
    'Lista de actividades: [MÚSICA ]': 12,
    'Lista de actividades: [MACRAMÉ]': 13,
    'Lista de actividades: [INFORMÁTICA]': 14,
    'Lista de actividades: [JARDINERÍA]': 15,
    'Lista de actividades: [MANUALIDADES]': 16,
    'Lista de actividades: [IDIOMAS]': 17, 
    'Lista de actividades: [COCINA]': 18,
    'Lista de actividades: [COCTELERÍA]': 19,
    'Lista de actividades: [CERVECERÍA ARTESANAL]': 20,
    'Lista de actividades: [CATAS DE COMIDA Y BEBIDA]': 21,
    'Lista de actividades: [BINGO]': 22, 
    'Lista de actividades: [PARCHIS]': 23,
    'Lista de actividades: [AJEDREZ]': 24, 
    'Lista de actividades: [TEATRO]': 25
    }
    df_new['itemId'] = df_new['itemId'].map(dict_activities)
    
    return df_new

df_algorithm = preprocess_form(df_cohousing)

    # Preprocessing
reader = Reader()
data = Dataset.load_from_df(df_algorithm, reader)

train, test = train_test_split(data, test_size=0.25)

#Trainning
svd = SVDpp()
SVD_model_for_pickle = svd.fit(train)
preds = svd.test(test)

#Trainning all data
trainfull = data.build_full_trainset()

svd = SVDpp()
SVD_model_for_pickle = svd.fit(trainfull)

SVD_model_for_pickle.predict(uid=1, iid=1)

def defiant_recommender(userId, dataframe, algorithm, n_recommendations, column_iid= None, column_uid= None):
    """
    This functions will use a trained algorithm to find the n top list of recommended items for a given userID.

    Parameters
    -----------

    userId (int): the user ID of the person that we want recommendations for.

    dataframe (object): the DataFrame containing three columns; userID, itemID and rating.

    algorithm (object): the trained algorith used to recommend items.

    n_rcommendations (int): the number of items recommended.

    column_iid (string): name of the column containing the item ID.

    column_uid (string): name of the column containing the user ID.


    return
    ------

    List of ID of items that an specific user will like.

    """
    item_ids = dataframe[column_iid].to_list()
    items_finished = dataframe[dataframe[column_uid] == userId][column_iid]

    items_no_finished = []
    for item in item_ids:
        if item not in items_finished:
            items_no_finished.append(item)

    preds = []
    for item in items_no_finished:
        preds.append(SVD_model_for_pickle.predict(uid=userId, iid=item))

    recommendations_rating = {pred[1]:pred[3] for pred in preds}

    order_dict = {k: v for k, v in sorted(recommendations_rating.items(), key=lambda item: item[1])}

    top_predictions = list(order_dict.keys())[:n_recommendations]
    
    return top_predictions

def check_recommended_item_name(list):
    """
    This functions will show the names of the n top rated items for a given userID.

    Parameters
    -----------

    list (object): the list of n recommended itemId.

    return
    ------

    A list with the n names of the itemId recommended to the given userId.

    """
    dict_items = {
            1: 'YOGA', 
            2: 'NATACION',
            3: 'BAILE', 
            4: 'GOLF',
            5: 'GIMNASIO',
            6: 'TIRO CON ARCO',
            7: 'ZUMBA', 
            8: 'TENIS',
            9: 'CLUB DE LECTURA',
            10: 'CLUB DE ESCRITURA',
            11: 'PINTURA', 
            12: 'MUSICA',
            13: 'MACRAME',
            14: 'INFORMATICA',
            15: 'JARDINERIA',
            16: 'MANUALIDADES',
            17: 'IDIOMAS', 
            18: 'COCINA',
            19: 'COCTELERIA',
            20: 'CERVECERIA ARTESANAL',
            21: 'CATAS DE COMIDA Y BEBIDA',
            22: 'BINGO', 
            23: 'PARCHIS',
            24: 'AJEDREZ', 
            25: 'TEATRO'
        }

    return [dict_items[i] for i in list]

def check_activities_user(userId, dataframe, n, column_rating= None, column_uid= None):
    """
    This functions will show the n top rated items for a given userID.

    Parameters
    -----------

    userId (int): the user ID of the person that we want recommendations for.

    dataframe (object): the DataFrame containing three columns; userID, itemID and rating.

    n (int): number of top rated items to show.

    column_rating (string): name of the column containing the item rating.

    column_uid (string): name of the column containing the user ID.


    return
    ------

    A dataframe with the n top rated items by that given user.

    """
    dataframe = dataframe[dataframe[column_uid] ==userId].sort_values(column_rating, ascending=False)[:n]
    
    #we create a dictionary to map the name of activities and change it for their activity code
    dict_activities = {
        1: 'YOGA', 
        2: 'NATACION',
        3: 'BAILE', 
        4: 'GOLF',
        5: 'GIMNASIO',
        6: 'TIRO CON ARCO',
        7: 'ZUMBA', 
        8: 'TENIS',
        9: 'CLUB DE LECTURA',
        10: 'CLUB DE ESCRITURA',
        11: 'PINTURA', 
        12: 'MUSICA',
        13: 'MACRAME',
        14: 'INFORMATICA',
        15: 'JARDINERIA',
        16: 'MANUALIDADES',
        17: 'IDIOMAS', 
        18: 'COCINA',
        19: 'COCTELERIA',
        20: 'CERVECERIA ARTESANAL',
        21: 'CATAS DE COMIDA Y BEBIDA',
        22: 'BINGO', 
        23: 'PARCHIS',
        24: 'AJEDREZ', 
        25: 'TEATRO'
    }

    dataframe['itemName'] = dataframe['itemId'].map(dict_activities)

    return dataframe
