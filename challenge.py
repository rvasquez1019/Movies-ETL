#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import psycopg2
import time


# In[2]:


# Establish reference files.
file_dir = r'C:\Users\chewbacca2019\Documents\Boot Camp\Week 8\Movies-ETL'

with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)
    
kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}/ratings.csv')


# In[3]:


def movies_etl(wiki_movies_raw, kaggle_metadata, ratings):
    


    # Select movies
    wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

   

    # Define clean_movie function
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                    alt_titles[key] = movie[key]
                    movie.pop(key)
        if len(alt_titles) > 0:
                movie['alt_titles'] = alt_titles
         
        
        # change column names
        def change_column_name(old_name, new_name):

            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        
            
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')
        

        return movie

    
    # Call clean_movies function to clean wiki_movies
    
    try:
        
        clean_movies = [clean_movie(movie) for movie in wiki_movies]
    
    except (NameError):
        
        print("Wrong Name. Please enter a defined name.")
    
    
    
    # Create wiki_movies_df
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    
    try:
        
        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
        
    except (KeyError,TypeError,NameError):
        
        print("imdb error")
    
    
    # Drop duplicated rows
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]   

        
    
    # Parse Box Office Data
    box_office = wiki_movies_df['Box office'].dropna() 
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    
    # Define a parse_dollars function
    def parse_dollars(s):
        
        try: 
            
            # if s is not a string, return NaN
            if type(s) != str:
                return np.nan

            # if input is of the form $###.# million
            if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

                # remove dollar sign and " million"
                s = re.sub('\$|\s|[a-zA-Z]','', s)

                # convert to float and multiply by a million
                value = float(s) * 10**6

                # return value
                return value

            # if input is of the form $###.# billion
            elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

                # remove dollar sign and " billion"
                s = re.sub('\$|\s|[a-zA-Z]','', s)

                # convert to float and multiply by a billion
                value = float(s) * 10**9

                # return value
                return value

            # if input is of the form $###,###,###
            elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

                # remove dollar sign and commas
                s = re.sub('\$|,','', s)

                # convert to float
                value = float(s)

                # return value
                return value

            # otherwise, return NaN
            else:
                return np.nan
        
        
        except (TypeError,ValueError):
            
            value = np.nan

    
    # Call parse_dollar function and extract values into a new column

    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        
    
    # Drop original uncleaned column
    wiki_movies_df.drop('Box office', axis=1, inplace=True)
    
    
    
    # Parse Budeget Data
    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    
    # Call parse_dollar function and extract values into a new column
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        
    # Drop original uncleaned column
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    
    
    
    # Parse Release Data
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'
    
    # Extract values from the above four forms and add into new column
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
    # Drop original uncleaned column
    wiki_movies_df.drop('Release date', axis=1, inplace=True)
    
    
    
    
    # Parse Running Time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    
    # Extract values
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    # Drop original uncleaned column
    wiki_movies_df.drop('Running time', axis=1, inplace=True)
    
    
    
    
    # Manage Kaggle Data
    
    # Remove bad data
    kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]
    # Remove adult movie data
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
    # Convert data type
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
    

    # Manage Rating Data
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    
    # Merge Wiki and Kaggle Metadata
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
    
    
    # Fill in missing data
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        
        try:
            
        
            df[kaggle_column] = df.apply(
                lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
                , axis=1)
            df.drop(columns=wiki_column, inplace=True)
        
        
        except (ValueError):
            pass
    
    
    # call fill_missing_kaggle_data to fill in zero values in kaggle_metadata
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
   

    
   # Drop Video columns
    movies_df.drop(columns=['video'], inplace=True)
    
    
    # Reorder Columns
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]
    
    # Rename Columns
    movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)
    
    
    
    # Manage Rating Data
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')
    
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    # Merge rating_counts with movie_df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    # Fills in zeros for movies without ratings
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
    
    
    
    # Load Data 
    db_string = f"postgres://postgres:{db_password}@localhost:5432/movie_data"
    engine = create_engine(db_string)
    
    
    
    # Delete all existing rows in movies table SQL
    rows_deleted = 0
    try:
        # read database configuration
        params = db_string
        # connect to the PostgreSQL database
        connection = psycopg2.connect(params)
        # create a new cursor
        cursor =  connection.cursor()

        # execute the UPDATE  statement
        Delete_all_rows = """truncate table movies """

        cursor.execute(Delete_all_rows)


        print("All Record Deleted successfully ")

        # get the number of updated rows
        rows_deleted = cursor.rowcount
        # Commit the changes to the database
        connection.commit()
        count = cursor.rowcount
        print (count, "Record deleted successfully in movies table")

    except (Exception, psycopg2.Error) as error :
        if(connection):
            print("Failed to delete row in movies table", error)
    finally:
        print("The movies data is loaded successfully")

   
    
    
    # Delete all rows in movies_ratings SQL table
    rows_deleted = 0
    try:
        # read database configuration
        params = db_string
        # connect to the PostgreSQL database
        connection = psycopg2.connect(params)
        # create a new cursor
        cursor =  connection.cursor()

        # execute the UPDATE  statement
        Delete_all_rows = """truncate table movies_ratings """

        cursor.execute(Delete_all_rows)


        print("All Record Deleted successfully ")

        # get the number of updated rows
        rows_deleted = cursor.rowcount
        # Commit the changes to the database
        connection.commit()
        count = cursor.rowcount
        print (count, "Record deleted successfully in movies_ratings table")

    except (Exception, psycopg2.Error) as error :
        if(connection):
            print("Failed to delet row in movies_ratings table", error)
    finally:
        print("The movie_ratings data is loaded successfully")

    
    # Load new dataframes into SQL
    
    movies_df.to_sql(name='movies', con=engine,if_exists='replace')
    
    movies_with_ratings_df.to_sql(name='movies_ratings', con=engine, if_exists='replace')
    
    
    
    
    # Delete all rows in ratings SQL table
    rows_deleted = 0
    try:
        # read database configuration
        params = db_string
        # connect to the PostgreSQL database
        connection = psycopg2.connect(params)
        # create a new cursor
        cursor =  connection.cursor()

        # execute the UPDATE  statement
        Delete_all_rows = """truncate table ratings """

        cursor.execute(Delete_all_rows)


        print("All Ratings Record Deleted successfully ")

        # get the number of updated rows
        rows_deleted = cursor.rowcount
        # Commit the changes to the database
        connection.commit()
        count = cursor.rowcount
        print (count, "Record deleted successfully in ratings table")

    except (Exception, psycopg2.Error) as error :
        if(connection):
            print("Failed to delete row in ratings table", error)
    finally:
        print("Start importing ratings data")
        
    
    # Import data to SQL ratings table
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        print(f'Done.{time.time() - start_time} total seconds elapsed')    
    
    
    print("ETL is finised")


# In[4]:


# Create rating counts for each movie so that we don't have to include every rating row.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()


# In[5]:


# Rename the userId column to count.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)


# In[6]:


# Pivot this data so that movieId is the index, columns will be rating values, and rows are rating counts.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[7]:


# Rename columns so they are easier to understand.
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[8]:


movies_etl(wiki_movies_raw, kaggle_metadata, ratings)


# In[ ]:




