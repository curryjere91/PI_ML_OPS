from fastapi import FastAPI
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
import scipy.sparse as sp


app = FastAPI()

#http://127.0.0.1:8000
df = pd.read_csv('clean_dataset.csv')

@app.get('/')
def index():   
    return {'Message from Jere to whoever reads it': 'Wisdom is the offsping of suffering and Time.'}


@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes: str):
    '''get all the movies produced on a certain month'''

    #dictionary to map english months to spanish.
    meses_ingles = {
        'January': 'enero',
        'February': 'febrero',
        'March': 'marzo',
        'April': 'abril',
        'May': 'mayo',
        'June': 'junio',
        'July': 'julio',
        'August': 'agosto',
        'September': 'septiembre',
        'October': 'octubre',
        'November': 'noviembre',
        'December': 'diciembre'
    }
    
    #mapping the dictionary
    meses_espanol = {v: k for k, v in meses_ingles.items()}
    
    #converting the column to datetime because for some reason sometimes it breaks
    df['release_date'] = pd.to_datetime(df['release_date'])

    #get month name and force it to lower case
    df['mes'] = df['release_date'].dt.month_name().str.lower()

    #return error if not using a month in spanish
    if mes.lower() not in meses_espanol:
        return {'error': 'Invalid month, enter a valid month in spanish'}
    
    #getting the spanish month, forced to lower case as well
    #filter month to the english month.
    df_mes = df[df['mes'] == meses_espanol[mes.lower()].lower()]
    nombre_mes = mes.capitalize()

    #get amount of movies for the month
    cantidad = len(df_mes)

    #return as dictionary
    return {'mes': nombre_mes, 'cantidad': cantidad}


@app.get('/peliculas_dia/{dia}')
def peliculas_dia(dia: str):
    '''get all the movies produced on a certain day'''
    
    #dictionary that maps the english days to their spanish counterparts
    dias_ingles = {
        'Monday': 'lunes',
        'Tuesday': 'martes',
        'Wednesday': 'miércoles',
        'Thursday': 'jueves',
        'Friday': 'viernes',
        'Saturday': 'sábado',
        'Sunday': 'domingo'
    }
    
    #mapping the dictionary
    dias_espanol = {unidecode(v): k for k, v in dias_ingles.items()}
        
    #converting the column to date time, just in case, again, same deal as with month
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    #getting the day of the week from the date
    df['day_of_week'] = df['release_date'].dt.day_name()

    #return error if input is not a day in spanish
    if unidecode(dia.lower()) not in dias_espanol:
        return {'error': 'Invalid day, enter a valid spanish day. Accents can be ignored'}

    # getting the Spanish day, forced to lower case and using unidecode to ignore accents
    df_dia = df[df['day_of_week'] == dias_espanol[unidecode(dia.lower())]]
    nombre_dia = dia.capitalize()

    #getting the amount of movies released on that day
    cantidad = len(df_dia)

    #returning it as a dictionary.
    return {'dia_semana': nombre_dia, 'cantidad': cantidad}


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''get the average and total earnings from a certain collection'''

    #filtering movies that belong to a collection the case and na exceptions are made to ignore the case from the input and avoid raising errors if there are empty values
    movie_collection = df[df['belongs_to_collection'].str.contains(franquicia, case=False, na=False)]

    if movie_collection.empty:
        return {'error': 'Movie collection not found'}

    #getting the amount of movies in that collection
    mov_quant= len(movie_collection)

    #adding together the total earnings from the movies, and also getting the average
    total_earnings = round(movie_collection['revenue'].sum(),2)
    avg_earnings = round(movie_collection['revenue'].mean(),2)

    #returning as dictioanry
    return {'franquicia': franquicia, 'cantidad': mov_quant, 'ganancia_total': f'{total_earnings:,}', 'ganancia_promedio': f'{avg_earnings:,}'}


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''get the amount of movies produced in a certain country'''
    
    #filtering movies by country
    countries = df[df['production_countries'].str.contains(pais, case=False, na=False)]

    if countries.empty:
        return {'error': 'Country not found'}

    #getting total movies produced within the country
    quant = len(countries)

    #return as dictionary
    return {'pais': pais.capitalize(), 'cantidad': quant}


@app.get('/productoras/{productora}')
def productoras(productora:str):
    '''get the amount of movies produced by a certain company'''

    #filtering the_production to include only rows where the productora is present in the production_companies column
    filtered_production =df[df['production_companies'].str.contains(productora, case=False, na=False)]

    if filtered_production.empty:
        return {'error': 'Production Company not found'}

    #calculating the total revenue and count the number of movies produced by the productora
    total_earnings = filtered_production['revenue'].sum()
    quant = filtered_production.shape[0]  #number of rows in the filtered_production

    #return as dictionary
    return {'productora': productora.title(), 'ganancia_total': f'{total_earnings:,}', 'cantidad': quant}


@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    '''get the investment, earnings and return on a certain movie'''

    #filtering df by specified movie
    filtered_movie = df[df['title'].str.contains(pelicula, case=False, na=False)]

    #if no movie is found, returns an error telling you that the movie isn't on the df
    if filtered_movie.empty:
        return {'error': 'Movie not found'}

    #just calling the correct columns for the filter.
    investment = filtered_movie['budget'].values[0]
    earnings = filtered_movie['revenue'].values[0]
    roi = round(filtered_movie['return'].values[0],2) #unclear if what was asked was ganancia - inversion or just to grab the return column, which is the more intuitive answer. 
    year = int(filtered_movie['release_year'].values[0])

    #return as dictionary
    return {'pelicula': pelicula.title(), 'inversion': f'{investment:,}', 'ganancia': f'{earnings:,}', 'retorno': f'{roi:,}', 'anio': year}



#ML

#filling the df with empty spaces to avoid errors, preprocessing the text, and then making the tfidf matrix for the recommendation function.

df.fillna({'overview': '', 'tagline': '', 'genres': '', 'belongs_to_collection': ''}, inplace=True)


def preprocess_text(text):
    '''this function grabs text, turns it to lower case, removes punctuation and numbers'''
    #lowercase the text
    text = text.lower()
    #remove punctuation
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text

# Preprocess the overview, tagline, and genres columns
df['preprocessed_text'] = df['overview'] + ' ' + df['tagline'] + ' ' + df['genres']  + ' ' + df['title'] + ' ' + df['belongs_to_collection']
df['processed_text'] = df['preprocessed_text'].map(preprocess_text)
df = df.drop(columns=['preprocessed_text'])

#calculate ifidf matrix. using TfidfVectorizer's list of stop words lets you skip some annoying processes of cleaning the text data. also makes it not eat all the ram
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['processed_text'])


@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    '''get recommended movies based on a certain movie. uses tfidf matrix to compare vectors from the dataset and the selected movie to generate recommendations'''
    movie = df[df['title'].str.contains(titulo, case=False, na=False)]

    if movie.empty:
        return {'error': 'Movie not found'}


    movie_index = movie.index[0]
    movie_vector = tfidf_matrix[movie_index]

    #calculating cosine similarity between the input movie and all other movies
    cosine_similarities = linear_kernel(movie_vector, tfidf_matrix).flatten()

    #getting the indices of movies sorted by similarity scores
    similar_movie_indices = cosine_similarities.argsort()[::-1]

    #filtering out the input movie itself
    similar_movie_indices = similar_movie_indices[similar_movie_indices != movie_index]

    #getting the top 5 recommendations
    similar_movie_indices = similar_movie_indices[:5]
    recommended_movies = list(df['title'].iloc[similar_movie_indices].str.title())

    #returning the recommended movies as dictionary
    return {'Lista recomendada': recommended_movies}


