# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:10:40 2024

@author: gopal
"""

import os
import sys

from pyspark import SparkConf, SparkContext
from operator import add

import re

def count_syllables(word):
    return len(
        re.findall('(?!e$)[aeiouy]+', word, re.I) +
        re.findall('^[^aeiouy]*e$', word, re.I)
    )

def count_personal_pronouns(text):
  pronoun_count = re.compile(r'\b(I|we|ours|my|mine|(?-i:us))\b', re.I)
  pronouns = pronoun_count.findall(text)
  return len(pronouns)


def Classify(x,positive,negative):

    if x in positive:
        return ['positive',1]
    
    elif x in negative:
        return ['negative',1]
    
    else:
        return ['neutal',1]

def Analyze(sentence, token, stopwords, positive, negative):
    
    try:
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        
        conf = SparkConf().setMaster("local").setAppName("analyze")
        sc = SparkContext(conf = conf)
        
        num_words = len(token)
        num_sent = len(sentence)
        
        token = sc.parallelize(token)
        sentence = sc.parallelize(sentence)
        
        personal_pronouns = sentence.map(lambda x : count_personal_pronouns(x)).sum()
      
        valid_token = token.filter(lambda x : x.lower() not in stopwords)
        num_cleaned_words = valid_token.count()
        
        avg_word_length = token.map(lambda x : len(x)).mean()
        avg_syllable_count = token.map(lambda x : count_syllables(x.lower())).mean()

        complex_words = token.map(lambda x : count_syllables(x.lower())).filter(lambda x : x > 2).count()

        type_dict = valid_token.map(lambda x : Classify(x.lower(),positive,negative)).reduceByKey(add)
        classified = type_dict.collect()
        
        positive_score = 0
        negative_score = 0
        
        for i in classified:
            if i[0] == 'positive':
                positive_score = i[1]
            elif i[0] == 'negative':
                negative_score = i[1]
        
        polarity_score = (positive_score - negative_score)/((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score)/(num_cleaned_words + 0.000001)
        
        avg_sentence_length = num_words/num_sent
        
        complex_words_percentage = complex_words/num_cleaned_words
        
        fog_index = 0.4*(avg_sentence_length + complex_words_percentage)
        
        sc.stop()
        
        return [positive_score,negative_score,polarity_score,subjectivity_score,avg_sentence_length,\
                complex_words_percentage,fog_index,avg_sentence_length,complex_words,num_cleaned_words,\
                avg_syllable_count,personal_pronouns,avg_word_length]
        
    except Exception as error:
        print(error)
        sc.stop()