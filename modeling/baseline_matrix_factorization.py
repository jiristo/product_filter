#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:04:51 2020

@author: jiristodulka
"""

import pandas as pd
import numpy as np
import gzip
import os

data_dir = os.path.join('/Users/jiristodulka/GoogleDrive/GitHub/product_filter','data')


toy = pd.DataFrame({'reviewerID': ['A','B','A','A', 'B','A','B','C','C','D'],
                    'item': ['m','n','d','p','m','d','l','l','k','m'],
                    'overall': [5,3,1,2,1,2,3,5,1,2]}
                   )


merged_df.describe(include = 'all')


for c in merged_df.columns:
    print(merged_df[c].value_counts())

ratings = merged_df[merged_df['style'] == {'Format:': ' DVD'}]
ratings.shape
ratings.columns
ratings['reviewer_rating_count'] = ratings.groupby('reviewerID')['overall'].transform('count')
ratings['title_rating_count'] = ratings.groupby('title')['overall'].transform('count')
ratings['verified'].value_counts(normalize = True)


ratings_subset = ratings[ (ratings['reviewer_rating_count'] > 20) & (ratings['title_rating_count']>20)]
ratings_subset = ratings_subset[ratings_subset['verified'] == True]
ratings_subset['overall'].value_counts(normalize = True)


ratings_subset_df = ratings_subset[['reviewerID', 'title', 'overall']]
ratings_subset_df.to_csv(data_dir + '/ratings_sample.csv', index = False)

toy.groupby('reviewerID')['overall'].transform('count')
