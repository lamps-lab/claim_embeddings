# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:00:58 2021

@author: weixi
"""

# This code is used to create 'score.csv' from the original 'SCORE_csv.xlsx' file.

# to load xlsx, need to 'pip install xlrd'
#Successfully installed xlrd-2.0.1
#XLRDError: Excel xlsx file; not supported
#not working

#pip uninstall xlrd
#pip install xlrd==1.2.0
# This works!!


import pandas as pd
import numpy as np
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence


path = r'SCORE_csv.xlsx'
claims = pd.read_excel(path, sheet_name = 0, usecols = ['DOI_CR', 'paper_id', 'coded_claim2','coded_claim3a','coded_claim3b', 'coded_claim4'], header=0)  #usecols = [0, 1]:choose column 5 and 9    #header=0: 

claims.to_csv('score.csv', encoding='utf-8', index=0)