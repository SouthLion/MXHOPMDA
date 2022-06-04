#!/usr/bin/env python
# coding: utf-8

# In[31]:


import csv
from tqdm import tqdm
import os
import pandas as pd


# In[6]:


path = os.getcwd()


# In[35]:


def process_data(path):
    path1 = path + '/All_Associations.csv'
    path2 = path + '/positive_associations.csv'
    paths = [path1,path2]
    datas = [[],[]]
    for p in paths:
        with open(p,'r',encoding = 'utf-8-sig') as f:
            lines = csv.reader(f)
            if 'pos' in p:
                for line in tqdm(lines):
                    datas[1].append(line[:2])
            else:
                for line in lines:
                    datas[0].append(line[:2])
    return datas


# In[36]:


all_data,pos_data = process_data(path)
print('the length of all_data:',len(all_data))


# In[37]:


def fun(all_data, pos_data):
    out = []
    for data in tqdm(all_data):
        if data in pos_data:
            out.append(data + [1])
        else:
            out.append(data + [0])
    return out


# In[38]:


final_data = fun(all_data,pos_data)


# In[29]:


def data_to_csv(data, title):
    pd_data = pd.DataFrame(columns=title, data=data)
    pd_data.to_csv('all_mirna_disease_associations.csv', encoding='utf-8', index = False)


# In[32]:


if __name__ == '__main__':
    data_to_csv(final_data,['miRNA','disease','label'])


# In[ ]:




