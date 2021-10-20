# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:00:58 2021

@author: weixi
"""


#pip install xlrd==1.2.0



import pandas as pd
import numpy as np



path = r'SCORE_CVD2_ta3.xlsx'
claims = pd.read_excel(path, sheet_name = 0, usecols = ['DOI_CR', 'ta3_pid', 'claim2_abstract', 'claim3_hyp', 'coded_claim4'], header=0)  #usecols = [0, 1]:choose column 5 and 9    #header=0: 

claims.to_csv('score_no_claim3b.csv', encoding='utf-8', index=0)
