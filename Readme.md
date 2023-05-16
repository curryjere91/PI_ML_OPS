

# <h1 align=center> **Individual Project Nº1 (MLOps)** </h1>


## Introduction
My take on the MLOps project for the Data Science course at Soy Henry.
This project leverages data from a public dataset on movies, containing information such as movie titles, genres, release dates, ratings, etc. 
In this project different techniques in ETL and (EDA) will be used in order to create a content-based movie recommendation system.
This model will then be deployed on Render using FastAPI to test it's online functionality.

## Objective

The main objective of this project is to develop and deploy a content-based movie recommendation system by leveraging data from a comprehensive dataset. 
As such, these are the main objectives:

- Apply ETL techniques to preprocess and clean the dataset.
- Conduct in-depth exploratory data analysis to gain insights into the dataset attributes in order to identify key features that could significantly influence the ML model.
- Develop a simple content-based machine learning model that uses similarity scores to filter and recommend movies based on user input.
- Design and implement a set of functions as solicited by the Soy Henry staff, and that integrates with the content-based recommendation system.
- Deploy the recommendation system within an API in a production environment, ensuring it is availability to users, taking into account the efficiency of the model due to the limited resources available in the selected environment in which the proyect is being deployed.

## Links:

- Data Preprocessing and analysis: [ETL & EDA](https://github.com/curryjere91/PI_ML_OPS/blob/main/ETL%20and%20EDA.ipynb)
- Functions and ML Development: [API Functions & recommendation system](https://github.com/curryjere91/PI_ML_OPS/blob/main/API%20functions%20and%20ML.ipynb)
- API deployment folder: [FastAPI config and deployment info](https://github.com/curryjere91/PI_ML_OPS/tree/main/fastapideployment)
- Link to deployed API: [Render](https://jereramipi-ml-ops.onrender.com/)
- Video of the API and code working(link if i actually do it): [Video presentation]()


# <h1 align=center> **Contents and details** </h1>

The project has been thoroughly explained within the Jupyter Notebook files, so this will be a quick summary of the thought process that went into the decision making and a quick summary instead. in the API Functions part there is a quick summary of what the functions are named, in case they are needed for the deployed API.

## ETL & EDA:

The data didn't really need much cleaning outside of the nested dictionaries, which were tricky to deal with, as some had json data.
But once the relevant data was extracted, it was very clear and easy to understand without any need for additional graphic aids or going deeper into the dataframe to get extra details; at least for what was solicited from it. Almost nothing about the dataset was noteworthy.
The only interesting case was popularity, that had some very extreme variances. But while this could have been exploited for something in the machine learning model, the project required us to drop the vote counts of each movie and just leave the scores. This completely ruined the possibility of exploring the relationship between vote counts, score and popularity of the movies to make a better recommendation algorithm, as with these 3 data points we could have reduced the ludicrous differences between popularity of the different movies. So that idea had to be discarded.

## API Functions

The first 6 functions are generic data retrieval queries based on the input. Here there is not much to talk about, they are pretty simple, as they only needed to take into account only basic data points and potential user input error, as well as using lower case, or accents, as some queries are done in spanish.
In case you want to use the functions in the deployed API, they must follow this guide:

Functions:

- peliculas_mes/month in spanish
Month must be in spanish, but it doesnt matter if it's lower case, or capitalized.
The query will return the amount of movies released in that month.

- peliculas_dia/day in spanish
Again, day must be in spanish, it doesn't matter if its lower case or capitalized, or if it has accents.
The query returns the amount of movies released on that day of the week.

- franquicia/collection(or movie franchise)
Here you input a movie collection, with the movie name in english.
It returns the total earnings and average of the entire franchise.

- peliculas_pais/country
Here you input a country name in english. The output will be the total of movies produced within that country.

- productoras/production company
Here you input a production company name, and you get the total earnings that company has had and the amount of movies it produced.

- retorno/movie name
here you input the movie name, and you get the budget, revenue and return it had, as well as the release year of the movie.

## Machine Learning
The machine learning model was fairly simple and straight forward. With the provided data, there really wasn't a lot that could be done, so the idea was to make a content-based filter.
The way it works is that it grabs the common words from the requested movie, it asigns them a numerical value, and then it compares that numerical value to that of the rest of the movies. The five movies with the closest numerical values to the movie that was looked up will be returned.
This is the short version, the more in depth explanation is within the Jupyter Notebook, that also has some weird cases, that couldn't really be classified as "edge cases", but just oddities that were found during testing.

To use this function in the API:
- recomendacion/movie name
Input the movie name, and you get a list of 5 movies.

## Extra stuff
+ The dataset that was used:[Dataset](https://drive.google.com/file/d/1Rp7SNuoRnmdoQMa5LWXuK4i7W1ILblYb/view?usp=sharing)
+ Additional information about the dataset:[Data dictionary](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit#gid=0)
+ Supporting material:[Link](hhttps://github.com/HX-PRomero/PI_ML_OPS/raw/main/Material%20de%20apoyo.md)
