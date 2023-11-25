import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, translate, to_date, split, explode
import s3fs

def playstore_transform():
    #create spark session
    spark = SparkSession.builder.appName('googleplaystore').getOrCreate()

    # load data using pandas (s3://bucket-name/folder/file.format)
    pd_store = pd.read_csv('s3://playstore-ratings-project/raw-data/googleplaystore.csv')
    pd_reviews = pd.read_csv('s3://playstore-ratings-project/raw-data/googleplaystore_user_reviews.csv')

    # convert them to pyspark df
    store = spark.createDataFrame(pd_store)
    reviews = spark.createDataFrame(pd_reviews)

    # created dim app df by selecting columns from store df
    dim_app = store.select(col("App"),col("Current Ver"),col("Android Ver"),col("Last Updated"),col("Size"),col("Content Rating"),col("Category"))
    # drop the duplicates in App column in app dimension 
    dim_app = dim_app.dropDuplicates(["App"])
    # trim the app name for extra spaces
    dim_app = dim_app.withColumn("App",trim(dim_app.App))
    # replace "(double quotes) with empty char in app name column
    dim_app = dim_app.withColumn("App",translate(col("App"),"\"",""))
    # cast Last Updated column to date, before it replace 0 with null
    dim_app = dim_app.replace(0,None,'Last Updated').withColumn("Last Updated",to_date(col("Last Updated"),"MMMM d, yyyy"))
    # convert it to pandas df and add index column in it
    pd_dim_app = dim_app.toPandas()
    pd_dim_app["App ID"] = pd_dim_app.index+1
    #convert to pandas datetime
    pd_dim_app['Last Updated'] = pd.to_datetime(pd_dim_app['Last Updated'], errors='coerce')

    # created dim category df by selecting columns from store df
    dim_category = store.select(col("Category"))
    # drop the duplicates in Category column in category dimension 
    dim_category = dim_category.dropDuplicates(["Category"])
    # convert it to pandas df and add index column in it
    pd_dim_category = dim_category.toPandas()
    pd_dim_category["Category ID"] = pd_dim_category.index+1

    # created dim genres df by selecting columns from store df
    dim_genres = store.select(col("App"),col("Genres"))
    # drop the duplicates in App column in genres dimension 
    dim_genres = dim_genres.dropDuplicates(["App"])
    # apply the same transformation for app name as applied earlier
    dim_genres = dim_genres.withColumn("App",translate(col("App"),"\"",""))
    dim_genres = dim_genres.withColumn("App",trim(dim_genres.App))
    # Split the Genres column with ';' and have a array of values in it
    dim_genres = dim_genres.withColumn("Genres",split(trim(dim_genres.Genres),";"))
    # Explode the Genres column, having a Row for each value present in the array
    dim_genres = dim_genres.withColumn("Genres",explode(dim_genres.Genres))
    # convert it to pandas df and add index column in it
    pd_dim_genres = dim_genres.toPandas()
    pd_dim_genres["Genres ID"] = pd_dim_genres.index+1

    # create dim date by droping duplicates in dates
    dim_date = dim_app.select(col('Last Updated'))
    dim_date = dim_date.dropDuplicates(["Last Updated"])
    #convert to pandas df and convert it to datetime
    pd_dim_date = dim_date.toPandas()
    pd_dim_date['Last Updated'] = pd.to_datetime(pd_dim_date['Last Updated'], errors='coerce')
    #add columns required to df
    pd_dim_date['Day'] = pd_dim_date['Last Updated'].dt.day
    pd_dim_date['Month'] = pd_dim_date['Last Updated'].dt.month
    pd_dim_date['Year'] = pd_dim_date['Last Updated'].dt.year
    pd_dim_date['Quarter'] = pd_dim_date['Last Updated'].dt.quarter
    pd_dim_date['Date'] = pd_dim_date['Last Updated']
    #drop NA rows
    pd_dim_date = pd_dim_date.dropna()
    #cast rows to int
    pd_dim_date['Day'] = pd_dim_date['Day'].astype(int)
    pd_dim_date['Month'] = pd_dim_date['Month'].astype(int)
    pd_dim_date['Year'] = pd_dim_date['Year'].astype(int)
    pd_dim_date['Quarter'] = pd_dim_date['Quarter'].astype(int)
    #create Date ID in YYYYMMDD format
    pd_dim_date['Date ID'] = pd_dim_date['Date'].astype(str)
    pd_dim_date['Date ID'] = pd_dim_date['Date ID'].str.slice(0,4)+pd_dim_date['Date ID'].str.slice(5,7)+pd_dim_date['Date ID'].str.slice(8,10)
    pd_dim_date['Date ID'] = pd_dim_date['Date ID'].astype(int)
    #selected required columns
    pd_dim_date = pd_dim_date[['Date ID','Date','Day','Month','Year','Quarter','Last Updated']]

    #create fact table using the pd dims
    pd_fact_apps = pd_dim_app.merge(pd_dim_category,on="Category").merge(pd_dim_genres,on="App").merge(pd_dim_date,on="Last Updated")
    #genearte an ID column
    pd_fact_apps["ID"] = pd_fact_apps.index+1
    #select the required columns
    pd_fact_apps = pd_fact_apps[['ID','App ID','Category ID','Date ID']]
    #rename the Date ID column to Last Update Date ID
    pd_fact_apps.rename({'Date ID':'Last Update Date ID'},axis = 1, inplace=True)
    #initialize new columns in fact with random numbers (just for test)
    pd_fact_apps['Installs'] = np.random.randint(1000,5000000, size=len(pd_fact_apps))
    pd_fact_apps['Reviews'] = np.random.randint(100,50000, size=len(pd_fact_apps))
    pd_fact_apps['Ratings'] = 0.0

    #select the required columns
    pd_dim_app = pd_dim_app[['App ID','App','Current Ver','Android Ver','Size','Content Rating','Last Updated']]
    #merge with dim app to get the App ID and select the required columns
    pd_dim_genres = pd_dim_genres.merge(pd_dim_app,on="App")[['Genres ID','App ID','Genres']]
    #select the required columns
    pd_dim_date = pd_dim_date[['Date ID','Date','Day','Month','Year','Quarter']]

    #create dim reviews table
    dim_reviews = reviews.select(col('App'),col('Translated_Review'))
    # apply the same transformation for app name as applied earlier
    dim_reviews = dim_reviews.withColumn("App",translate(col("App"),"\"",""))
    dim_reviews = dim_reviews.withColumn("App",trim(dim_reviews.App))
    dim_reviews = dim_reviews.withColumnRenamed('Translated_Review','Review')
    pd_dim_reviews = dim_reviews.toPandas()
    #generate index ID
    pd_dim_reviews["Review ID"] = pd_dim_reviews.index+1
    #initialize new columns in fact with random numbers (just for test)
    pd_dim_reviews['Sentiment Analysis'] = np.random.uniform(-5.0,5.0,len(pd_dim_reviews))
    pd_dim_reviews['Ratings'] = np.random.randint(1,5, size=len(pd_dim_reviews))
    #merge with dim app to get the App ID and select the required columns
    pd_dim_reviews = pd_dim_reviews.merge(pd_dim_app,on="App")[['Review ID','App ID','Review','Ratings','Sentiment Analysis']]

    #create dim installs table
    dim_installs = reviews.select(col('App'))
    # apply the same transformation for app name as applied earlier
    dim_installs = dim_installs.withColumn("App",translate(col("App"),"\"",""))
    dim_installs = dim_installs.withColumn("App",trim(dim_installs.App))
    pd_dim_installs = dim_installs.toPandas()
    #generate index ID
    pd_dim_installs["Install ID"] = pd_dim_installs.index+1
    #initialize new columns in fact with random numbers (just for test)
    pd_dim_installs['User'] = np.random.randint(1,5000000,size=len(pd_dim_installs))
    pd_dim_installs['Device'] = np.random.randint(1,70000000, size=len(pd_dim_installs))
    #merge with dim app to get the App ID and select the required columns
    pd_dim_installs = pd_dim_installs.merge(pd_dim_app,on="App")[['Install ID','App ID','User','Device']]
    
    #write files in S3 bucket (s3://bucket-name/folder/file.format)
    pd_dim_app.to_csv("s3://playstore-ratings-project/transformed-data/dim_app/dim_app.csv",index=False)
    pd_dim_category.to_csv("s3://playstore-ratings-project/transformed-data/dim_category/dim_category.csv",index=False)
    pd_dim_genres.to_csv("s3://playstore-ratings-project/transformed-data/dim_genres/dim_genres.csv",index=False)
    pd_dim_date.to_csv("s3://playstore-ratings-project/transformed-data/dim_date/dim_date.csv",index=False)
    pd_dim_reviews.to_csv("s3://playstore-ratings-project/transformed-data/dim_reviews/dim_reviews.csv",index=False)
    pd_dim_installs.to_csv("s3://playstore-ratings-project/transformed-data/dim_installs/dim_installs.csv",index=False)
    pd_fact_apps.to_csv("s3://playstore-ratings-project/transformed-data/fact_apps/fact_apps.csv",index=False)
