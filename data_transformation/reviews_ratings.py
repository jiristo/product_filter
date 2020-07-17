#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:55:16 2020

@author: jiristodulka
"""

import pandas as pd
import numpy as np
import gzip
import os
import json
import matplotlib 
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
import string

import textblob
from textblob import TextBlob


data_dir = os.path.join('/Users/jiristodulka/GoogleDrive/GitHub/product_filter','data')



def load_meta():
    '''
    Desc.: 
        - loads metadata that sore info about 'Movies & TV' ONLY
    Input:
        - By Default it takes the data from ./data directory
    Returns:
        - meta_df : pd.DataFrame  object with info about items
        
    '''
    meta = []
    with gzip.open(data_dir +'/meta_Movies_and_TV.json.gz') as f:
        for l in f:
            meta.append(json.loads(l.strip()))
            
    meta_df = pd.DataFrame(meta)[['main_cat', 'title','asin']]
    meta_df = meta_df[meta_df['main_cat']== 'Movies & TV']
    return meta_df




def load_reviews():
    '''
    Desc.:
        - Load Reviews
    Input:
        - By Default it takes the data from ./data directory
    Returns:
        reviews_df: pd.DataFrame object storing ALL the reviews for MULTIPLE CATEGORIES
        
    '''
    reviews = []
    for line in open(data_dir + '/Movies_and_TV_5.json', 'r'):
        reviews.append(json.loads(line))
    
    reviews_df = pd.DataFrame(reviews)
    return reviews_df




def merge_reviews_meta(reviews_df, meta_df):
    '''
    
    
    
    '''
    merged_df = pd.merge(reviews_df, meta_df[['title', 'asin']],
                         how = 'inner', left_on='asin', right_on = 'asin')
    merged_df['char_count'] = merged_df['reviewText'].str.len()
    #merged_df = merged_df[merged_df['char_count']<200] 
    return merged_df





def downsample_reviews(merged_df, rating_min = 10 ,length = [300,800]):
    '''
    Desc.: 
        - Subsets the merged_df input to extract only relevant records ("Movies and TV"):
            1. selects only movies category
            2. N/A
            3. length of reviews in certain range
            4. only certain # of reviews
    Input:
       - merged_df: output of merge_reviews_meta(reviews_df, meta_df)
       - length: min and max length of reviews in range
       - trashold: max number of reviews per movie
       
     Returns:
         downsampled reviews pd.DataFrame
    '''
    down_reviews_df = merged_df.copy()
    
    down_reviews_df['char_count'] = down_reviews_df['reviewText'].str.len()
    down_reviews_df['sum_reviews'] = down_reviews_df.groupby('title')['title'].transform('count')

    sample = down_reviews_df[down_reviews_df['char_count'].between(length[0], length[1])]
    sample =  sample[sample['sum_reviews'] >= rating_min]
    
    titles_index = sample.title.value_counts()[sample.title.value_counts()>=rating_min].index 
    sample = sample[sample['title'].isin(titles_index)]
    
    sample_df = sample.groupby('title').apply(lambda x: x.sample(rating_min)).reset_index(drop = True)

    return sample_df


# sample_df['sum_reviews'].min()
# sample_df['sum_reviews'].value_counts().value_counts()[:50]
# sample_df.sample(n=50).groupby('title')['title'].value_counts().value_counts(normalize = True)
# sample_df.groupby('title').apply(lambda x: x.sample(n=50)).reset_index(drop = True)
# sample_df.title.value_counts()
# sample_df['sum_reviews'].min()
# sample_df['char_count'].max()
# sample_df['char_count'].min()

def clean_reviews(sample_df):
    '''
    Desc.:
        Clean 'reviewText', extracts adjectives for each review into a list in new column: review_adjectives
    Input:
        - sample_df: pd.DataFrame as sampled reviews
    Output:
        - ...: identical to the input but with new columns storing the adjectives in review's in the  a list
    
    '''
    clean_sample = sample_df.copy()
    clean_sample['reviewText']=clean_sample.reviewText.str.lower()
    clean_sample['reviewText'] = clean_sample['reviewText'].str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
    
    def get_adjectives(text):
        blob = TextBlob(text)
        '''
        Extracts adjectives
        '''
        return [ word for (word,tag) in blob.tags if tag == "JJ"]
    
    clean_sample_df = clean_sample.copy()
    clean_sample['review_adjectives'] = clean_sample['reviewText'].apply(get_adjectives)
    clean_sample_df = clean_sample.copy()
    return clean_sample_df

clean_sample_df = clean_reviews(sample_df)

# boo = sample_df.loc[:,['title', 'reviewText']]
# boo['reviewText']=sample_df.reviewText.str.lower()
# boo['reviewText'] = boo['reviewText'].str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
# boo['split_words'] = [ nltk.word_tokenize( str(boo['reviewText']) ) for sentence in boo['reviewText'] ]

# def get_adjectives(text):
#     blob = TextBlob(text)
#     return [ word for (word,tag) in blob.tags if tag == "JJ"]

# boo['review_adjectives'] = boo['reviewText'].apply(get_adjectives)

# print(splitwords)
# stop = set(stopwords.words('english'))
# exclude = set(string.punctuation)


def main():
    meta_df = load_meta()
    
    reviews_df = load_reviews()
    
    merged_df = merge_reviews_meta(reviews_df, meta_df)
    
    sample_df = downsample_reviews(merged_df)
    
    clean_sample_df =  clean_reviews(sample_df)
    
    return clean_sample_df, merged_df

if __name__ == "__main__":
    clean_sample_df, merged_df = main() 
    
    
    # clean_sample_df.to_csv(data_dir + '/sampled_reviews.csv', index = False)

    
    
    
    
    
    
    