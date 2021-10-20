# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:23:49 2021

@author: weixin
"""

#This code is used to create embeddings for claim sentences
# language model used here is RoBERTa
# the embeddings for the 4 claim sentences are averaged to create the final embedding

# need to run under flair environment, which can be created by the commands below:
#conda create -n flair python=3.6
#conda activate flair
#pip install flair



import pandas as pd
#from pandas.core.frame import DataFrame
import numpy as np
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence



claims = pd.read_csv(r'score.csv', encoding="utf8", usecols=['DOI_CR', 'paper_id', 'coded_claim2','coded_claim3a','coded_claim3b', 'coded_claim4']) 

print(claims)
claims_list = claims.values.tolist()




# init embedding
#embedding = TransformerWordEmbeddings('xlnet-base-cased')
#embedding = TransformerWordEmbeddings('distilbert-base-cased')
#embedding = TransformerWordEmbeddings('bert-base-uncased')
embedding = TransformerWordEmbeddings('roberta-base')
print('====================================model loaded!')

n=0
all_average_embb = np.zeros(768)
all_average_embb = np.append('DOI_CR', all_average_embb)
all_average_embb = np.append('paper_id', all_average_embb)
for i in range(0, len(claims_list)):
    n+=1
    print(n)
    input_s_all = claims_list[i]
    DOI_CR = input_s_all[0]
    paper_id = input_s_all[1]
    input_s = input_s_all[2:]
    
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
    
    
 
    
    
    #print("average***************")
    average = (sentence[0].embedding +sentence[1].embedding +sentence[2].embedding +sentence[3].embedding)/4
    
    #print(average)
    #print(average.size())
    
    average = average.data.cpu().numpy()  # convert tensor format into list
    average = np.append(DOI_CR, average)
    average = np.append(paper_id, average)
    
    #print(average)
    

    
    all_average_embb = np.vstack((all_average_embb, average))     # append a list to another vertically
    
    
    
    
np.savetxt('emb_roberta_average.csv',all_average_embb,delimiter=',', fmt="%s")
