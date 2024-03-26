# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:31:29 2024

@author: gopal
"""

import os
import sys

from pyspark import SparkConf, SparkContext



def Stop_Words():

    try:
  
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
        conf = SparkConf().setMaster("local").setAppName("Stop_words")
        sc = SparkContext(conf = conf)
        path = r"StopWords"
        files = os.listdir(path)
      
        stopwords = []
        for file in files:
            rdd = sc.textFile(r"StopWords\{}".format(file))
         
            words = rdd.map(lambda x : x.split())
            words = rdd.map(lambda x : str(x.split('|')[0].strip()).lower())
            stopwords.extend(words.collect())
            
        sc.stop()

        return stopwords
    
    except():
        
        sc.stop()
        return 'error'



#%%
