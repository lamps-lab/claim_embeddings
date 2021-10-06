# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:23:49 2021

@author: weixin
"""

#This code is used to create embeddings for claim sentences
# language model used here is RoBERTa




import pandas as pd
#from pandas.core.frame import DataFrame
import numpy as np
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence



claims = pd.read_csv(r'score.csv', encoding="utf8", usecols=['DOI_CR', 'paper_id', 'coded_claim2','coded_claim3a','coded_claim3b', 'coded_claim4']) 
#claims = pd.DataFrame(df) 
print(claims)
claims_list = claims.values.tolist()

#claims2 = df['coded_claim2']
#claims3 = df['coded_claim3a']


# init embedding
#embedding = TransformerWordEmbeddings('bert-base-uncased')
embedding = TransformerWordEmbeddings('roberta-base')
print('====================================model loaded!')

n=0
all_maxpool_embb = np.zeros(768)
all_maxpool_embb = np.append('DOI_CR', all_maxpool_embb)
all_maxpool_embb = np.append('paper_id', all_maxpool_embb)
for i in range(0, len(claims_list)):
    n+=1
    print(n)
    input_s_all = claims_list[i]
    DOI_CR = input_s_all[0]
    paper_id = input_s_all[1]
    input_s = input_s_all[2:]
    #input_s = 'The grass is green.'
    #print("***************", input_s)
    #print(DOI_CR)
    #print(paper_id)
    

    
    
    # create a sentence
    sentence = Sentence(input_s)
    
    # embed words in sentence
    embedding.embed(sentence)
    
    #print("0***************")
    #print(sentence[0].embedding)
    #print(sentence[0].embedding.size())
    
    #print("1***************")
    #print(sentence[1].embedding)
    #print(sentence[1].embedding.size())
    
    #print("2***************")
    #print(sentence[2].embedding)
    #print(sentence[2].embedding.size())
    
    #print("3***************")
    #print(sentence[3].embedding)
    #print(sentence[3].embedding.size())
    
    
    s0 = sentence[0].embedding.data.cpu().numpy()
    s1 = sentence[1].embedding.data.cpu().numpy()
    s2 = sentence[2].embedding.data.cpu().numpy()
    s3 = sentence[3].embedding.data.cpu().numpy()
    
    maxpool = np.zeros(768)
    for i in range(0, len(s0)):
        list= [s0[i], s1[i], s2[i], s3[i]]
        maxvalue = max(list)
        maxpool[i]= maxvalue
    
    
    maxpool = np.append(DOI_CR, maxpool)
    maxpool = np.append(paper_id, maxpool)
    
    #print(average)
    

    
    all_maxpool_embb = np.vstack((all_maxpool_embb, maxpool))     # append a list to another vertically
    
    
    
    
np.savetxt('emb_roberta_maxpool.csv',all_maxpool_embb,delimiter=',', fmt="%s")
