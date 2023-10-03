#importación de librerias
from fastapi import FastAPI
from typing import Union
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity

# Desactivación de warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


# instancio
app = FastAPI()
#http://127.0.0.1:8000

#archivo
combinado = pd.read_csv('C:\\Users\\pc\\Desktop\\FastAPI\\final.csv')
ml= pd.read_csv('C:\\Users\\pc\\Desktop\\FastAPI\\ML.csv')

#decorador donde se le indica que cuando llegue a
#ruta madre muestre el siguiente mensaje
@app.get('/')
def index():
    return{'Trabajo individual de Camila Duran'}




#comienzo con funciones

#def PlayTimeGenre( genero: str ) : Debe devolver añocon más horas jugadas para dicho género.
#Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
#http://127.0.0.1:8000/PlayTimeGenre/Action/

@app.get('/PlayTimeGenre/{genero}/')
def PlayTimeGenre(genero: str):
    resultado = combinado

    # Filtra el DataFrame por género
    Filtro= resultado["genres"] == genero


    # Calcula el total de horas jugadas por año
    horas_por_año = (resultado[Filtro].groupby(['year']).sum()[["playtime_forever"]]
                     .sort_values(by='playtime_forever', ascending = False).reset_index())
    

    retorno = {f"Año de lanzamiento con más horas jugadas para Género {genero.title()}": str(int(horas_por_año.iloc[0]['year']))}


    #control error de consulta
   
    return retorno

    # Devuelve un diccionario con el año y las horas jugadas
    #return {"Año de lanzamiento con más horas jugadas para Género {}": año_con_mas_horas}


#def UserForGenre( genero: str ) : Debe devolver el usuario que acumula más horas jugadas 
# para el género dado y una lista de la acumulación de horas jugadas por año.

#http://127.0.0.1:8000/UserForGenre/Action/

@app.get('/UserForGenre/{genero}/')
def UserForGenre(genero:str):
    # Filter the DataFrame by genre
    fil = combinado["genres"] == genero
    
    # Filter the DataFrame to include only the specified genre
    genre_data = combinado[fil]

    # Find the user with the most playtime in the specified genre
    Usu_maxhorajugadas = genre_data.groupby(['user_id']).sum()['playtime_forever'].idxmax()

    # Group the data by year and calculate total playtime for each year
    año_horas = genre_data.groupby('year')['playtime_forever'].sum().reset_index()

    # Convert año_horas DataFrame to a list of dictionaries
    horas_jugadas = año_horas.to_dict(orient='records')

    # Return the results as a dictionary
    return {
        "Usuario con más horas jugadas para Género {}".format(genero): Usu_maxhorajugadas,
        "Horas jugadas por año": horas_jugadas
    }


#def UsersRecommend( año: int ) : Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True 
# y comentarios positivos/neutrales)
#http://127.0.0.1:8000/UsersRecommend/2015/

@app.get("/UsersRecommend/{anio}/")
async def UsersRecommend(anio: int):
    # Filtrar los datos para el año especificado y donde recommend es True
    data_filt = combinado[(combinado['year_posted'] == anio) & (combinado['recommend'] == True) & (combinado['sentiment_analysis'] >= 1)]
    
    #ordenar por recommend
    orden_data= data_filt.sort_values(by='sentiment_analysis', ascending=False)

    #top3
    top_3= orden_data.head(3)

    respuesta= [
        {'Puesto uno': top_3.iloc[0]['title']},
        {'Puesto dos': top_3.iloc[1]['title']},
        {'Puesto tres': top_3.iloc[2]['title']}

    ]
    return respuesta
    

#def UsersNotRecommend( año: int ) : Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
#  (reviews.recommend = False y comentarios negativos) Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
#http://127.0.0.1:8000/UsersNotRecommend/2015/

@app.get('/UsersNotRecommend/{anio}/')
async def UsersNotRecommend(anio: int ):
    # Filtrar los datos para el año especificado y donde recommend es False
    fil_data= combinado[(combinado['year_posted'] == anio) & (combinado['recommend'] == False) & (combinado['sentiment_analysis'] ==0)]

    #orden por recommend
    dat_ord = fil_data.sort_values(by='sentiment_analysis', ascending=False)

    #top 3
    top= dat_ord.head(3)

    resp= [
          {'Puesto uno': top.iloc[0]['title']},
        {'Puesto dos': top.iloc[1]['title']},
        {'Puesto tres': top.iloc[2]['title']}

    ]
    return resp


#def sentiment_analysis( año: int ) : Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentran categorizados con un análisis de sentimiento.
#Ejemplo de retorno: {Negativo = 182, Neutral = 120, Positivo = 278}
#http://127.0.0.1:8000/sentiment_analysis/2015/

@app.get('/sentiment_analysis/{anio}/')
async def sentiment_analysis(anio : int):

    #filtrar por año
    filtro_anio=combinado[combinado['year'] == anio]

    #conteo de analisis de sentimiento
    sentimientos= filtro_anio['sentiment_analysis'].value_counts().to_dict()

    #diccionario
    dic= {
        'Negativo': sentimientos.get(0,0),
        'Neutral': sentimientos.get(1,0),
        'Positivo' : sentimientos.get(2,0)

    }
    return dic






#ML
#http://127.0.0.1:8000/recomendacion_juego/282010/

@app.get('/recomendacion_juego/{id_producto}/')
async def recomendacion_juego(id_producto:int):
        # Verificar y convertir el tipo de datos de id_producto
        item_id_dtype = ml['item_id'].dtype
        if not isinstance(id_producto, type(item_id_dtype)):
            id_producto = item_id_dtype.type(id_producto)

        # Calcular características del juego (puedes expandir esto)
        game_features = ml.groupby('title').agg({'recommend_int': 'mean'})

        game_similarity = cosine_similarity(game_features)

        # Obtener el índice del juego de entrada en la matriz de similitud
        idx = ml [ml['item_id'] == id_producto].index[0]
    
        # Obtener las similitudes entre el juego de entrada y todos los juegos
        similarities = game_similarity[idx]

        # Ordenar los juegos por similitud y obtener los 5 juegos más similares
        similar_games_indices = similarities.argsort()[::-1][1:6]
    
        # Obtener los nombres de los juegos recomendados
        recommended_games = ml.iloc[similar_games_indices]['title'].tolist()
    
        return recommended_games








