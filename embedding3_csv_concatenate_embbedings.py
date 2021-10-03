# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:23:49 2021

@author: weixi
"""

#This code is used to create embeddings for claim sentences
# language model used here is RoBERTa
# the embeddings for the 4 claim sentences are concatenated to create the final embedding

# need to run under flair environment, which can be created by the commands below:
#conda create -n flair python=3.6
#conda activate flair
#pip install flair


import pandas as pd
#from pandas.core.frame import DataFrame
import numpy as np
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence



claims = pd.read_csv(r'score.csv', usecols=['DOI_CR', 'paper_id', 'coded_claim2','coded_claim3a','coded_claim3b', 'coded_claim4']) 

print(claims)
claims_list = claims.values.tolist()




# init embedding
#embedding = TransformerWordEmbeddings('bert-base-uncased')
embedding = TransformerWordEmbeddings('roberta-base')
print('====================================model loaded!')

n=0
all_concatenate_embb = np.zeros(3072)
all_concatenate_embb = np.append('DOI_CR', all_concatenate_embb)
all_concatenate_embb = np.append('paper_id', all_concatenate_embb)
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
    
    
 
    
    
    s0 = sentence[0].embedding
    s0 = s0.data.cpu().numpy()  # convert tensor format into list
    s1 = sentence[1].embedding
    s1 = s1.data.cpu().numpy()  # convert tensor format into list
    s2 = sentence[2].embedding
    s2 = s2.data.cpu().numpy()  # convert tensor format into list
    s3 = sentence[3].embedding
    s3 = s3.data.cpu().numpy()  # convert tensor format into list
    
    
    s01 = np.append(s0, s1)
    s012 = np.append(s01, s2)
    s0123 = np.append(s012, s3)
    
 
    s0123 = np.append(DOI_CR, s0123)
    s0123 = np.append(paper_id, s0123)
    
    
    
    
    all_concatenate_embb = np.vstack((all_concatenate_embb, s0123))     # append a list to another vertically
    
    
    
    
np.savetxt('emb_roberta_concatenate.csv',all_concatenate_embb,delimiter=',', fmt="%s")    
