import os, re
import json
import torch
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def convert (camel_input):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return ' '.join(map(str.lower, words))

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
    headers = convert(headers)
    headers = headers.replace('_',' ')
    headers = headers.replace('-',' ')
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
       pass
    return result


def main():

    parser = argparse.ArgumentParser(description='Process WikiTables corpus')
    parser.add_argument('-i', '--input', default='/raid/wake/data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='stb', choices=['stb', 'rbt', 'brt','fst','w2v','blo', 'sci'],
                        help='Model to use: "sbt" (Sentence-BERT, Default), "rbt" (Roberta),"fst" (fastText), "w2v"(Word2Vec), "blo" (Bloomer), "sci" sci-bert '
                             ' "brt" (Bert)')
    parser.add_argument('-r', '--result', default='./results', help='Name of the output folder that stores the similarity values calculated')

    args = parser.parse_args()

    files_path = args.input
    files = os.listdir(files_path)

    ignored = 0
    similarities = []
    for file in tqdm(files):
        if file[-4:] == 'json':
            try:
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
                    if len(df.index)>1000:
                        df = df.sample(frac=0.1, replace=True, random_state=1)

                    for col in df.columns:
                        aux = proccessText(' '.join(df[col].astype(str).tolist()))

                        emb = []
                        if len(aux) > 100: # Comprobamos si el tamaño de batc va a ser muy grande
                            for i in range(0,len(aux),100):
                                emb+= getEmbeddings(aux[i:i+100])
                        else:
                            emb = getEmbeddings(aux)

                        if emb != []:
                            aux = np.mean(emb, axis=0)
                            t2.append(aux)
                    
                    t2 = np.mean(t2, axis=0)
        
                    similarities.append(getSimilarity(a, t1, t2, data['id']))
            except Exception as e:
                print(e)
                print(a)
                print(t1)
                ignored+=1

                
    print('Ignored:', ignored)
    # Create the pandas DataFrame
    df_final = pd.DataFrame(similarities, columns = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    
    # print dataframe
    df_final.to_csv(os.path.join(args.result, args.model + '_alpha_similarity.csv'), index=False)

if __name__ == "__main__":
    main()