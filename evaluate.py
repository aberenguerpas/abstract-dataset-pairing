import os
import json
import torch
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

def getEmbeddings(data):

    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})

    if response.status_code == 200:
        return response.json()['emb']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod )


def proccessText(text):
    # Lowercase the text because the sentence model is uncased
    text = text.replace('&nbsp;',' ')
    text = text.lower()

    # The maximum token length admitted by is 256
    max_sequence_length = 256
    # Larger texts are cut into 256 token length pieces and their vectors are averaged
    # Take into account that a cell could have more than one token
    if len(text.split()) > max_sequence_length:
        list_tokens = text.split()
        list_texts = []
        for i in range(0, len(text.split()), max_sequence_length):
            list_texts.append(' '.join(list_tokens[i:i+max_sequence_length]))
        return list_texts
    else:
        return [text]

def proccessHeaders(headers):
    # Lowercase the text because the sentence model is uncased
    headers = [ h if not 'Unnamed:' in h else '' for h in headers]
    headers = ' '.join(headers)
    headers = headers.lower()
    headers = headers.replace('&nbsp;',' ')
    # The maximum token length admitted by is 256
    max_sequence_length = 256
    # Larger texts are cut into 256 token length pieces and their vectors are averaged
    # Take into account that a cell could have more than one token
    if len(headers.split()) > max_sequence_length:
        list_tokens = headers.split()
        list_texts = []
        for i in range(0, len(list_tokens), max_sequence_length):
            list_texts.append(' '.join(list_tokens[i:i+max_sequence_length]))
        return list_texts
    else:
        return [headers]


def getSimilarity(a, t1, t2, id):
    result = []
    cos = torch.nn.CosineSimilarity(dim=0)
    try:
        for alpha in np.arange(0, 1.1, 0.1).tolist():
            sim = cos(torch.from_numpy(a), torch.from_numpy(t1))*alpha + cos(torch.from_numpy(a), torch.from_numpy(t2))*(1-alpha)
            result.append(float(sim))
    except Exception as e:
        print(e)
        print(id)
        print(a)
 

    return result


def main():

    files_path = './data/'
    files = os.listdir(files_path)

    similarities = []
    for file in tqdm(files):
        if file[-4:] == 'json':
            with open(files_path+file, 'r') as f:
                data = json.load(f)

                # Obtenemos el embedding del abstract 'a'
                a = proccessText(data['desc'])
          
                a = np.mean(getEmbeddings(a), axis=0)
                
                # Obtenemos el embedding de su dataset t t1(header) t2(content)
                df = pd.read_csv(files_path+str(data['id'])+'.csv', encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None)
                t1 = proccessHeaders(df.columns.values)
                t1 = np.mean(getEmbeddings(t1), axis=0)
                
                t2 = []
                if len(df.columns)>1000:
                    continue
                if len(df.index)>100 and len(df.index)<10.000:
                    df = df.sample(frac=0.05, replace=True, random_state=1)
                if len(df.index)>=10000:
                    df = df.sample(n=1000, replace=True, random_state=1)
                
                for col in df.columns:
                    aux = proccessText(' '.join(df[col].astype(str).tolist()))
                    emb = getEmbeddings(aux)
                    if emb != []:
                        aux = np.mean(emb, axis=0)
                        t2.append(aux)
                
                t2 = np.mean(t2, axis=0)
     

                similarities.append(getSimilarity(a, t1, t2, data['id']))

    # Create the pandas DataFrame
    df_final = pd.DataFrame(similarities, columns = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    
    # print dataframe
    df_final.to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()