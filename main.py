# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:09:18 2024

@author: gopal
"""

import requests
from bs4 import BeautifulSoup

import os
import sys
import csv

from nltk.tokenize import word_tokenize,sent_tokenize

from stop_words import Stop_Words
from analyze import Analyze

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


spark = SparkSession.builder.appName("Assignment").getOrCreate()

schema = StructType([ StructField("URL_ID", StringType(), False), \
                     StructField("URL", StringType(), False), ])
    
        
input_urls = spark.read.schema(schema).options(header = True).csv(r"Input.csv")

input_urls = input_urls.filter('URL is not NULL')
url = input_urls.select(["URL","URL_ID"]).collect()

spark.stop()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
 
conf = SparkConf().setMaster("local").setAppName("main")
sc = SparkContext(conf = conf)

positive = sc.textFile(r"MasterDictionary\positive-words.txt")
negative = sc.textFile(r"MasterDictionary\negative-words.txt")
 
positive = positive.map(lambda x : str(x.split()[0].strip()).lower()).collect()
negative = negative.map(lambda x : str(x.split()[0].strip()).lower()).collect()

sc.stop()

stopwords = Stop_Words() 


for i in url:
    
    output = [i[0],i[1]]
    arr = []
    url = i[0]
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string
    
    if title != "Page not found - Blackcoffer Insights":
        print(title)
        try:
            content = soup.find("div", class_="td-post-content tagdiv-type").text
        except:
            content = soup.find_all("div",class_="tdb-block-inner td-fix-index")[14].text
            
       
        tokens = word_tokenize(content)
        tokens = list(filter(lambda x : x.isalpha(),tokens))
        sentence = sent_tokenize(content)
        
        arr = Analyze(sentence, tokens, stopwords, positive, negative)
    else:
        continue
    output.extend(arr)
    
    file = open(r"output.csv", 'a+', newline ='')
    
    with file:
        print('success')
        write = csv.writer(file)
        write.writerow(output)
    
    
    
    


    
    
        
   
#%%
