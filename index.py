import os, json
import faiss
import numpy as np
from tqdm import tqdm
import requests
from utils import *
import pandas as pd
import traceback
import argparse


def getEmbeddings(data):
    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})
    if response.status_code == 200:
        return response.json()['emb']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod)


def proccessHeaders(headers):
    # Lowercase the text because the sentence model is uncased
    headers = [ h if not 'Unnamed:' in h else '' for h in headers]
    headers = ' '.join(headers)
    headers = headers.lower()
    headers = headers.replace('&nbsp;',' ')
    # The maximum token length admitted by is 256
    max_sequence_length = 128
    # Larger texts are cut into 256 token length pieces and their vectors are averaged
    # Take into account that a cell could have more than one token
    if len(headers) > max_sequence_length:
        list_texts = []
        for i in range(0, len(headers), max_sequence_length):
            list_texts.append(headers[i:i+max_sequence_length])
        return list_texts
    else:
        return [headers]


def proccessText(text):
    # Lowercase the text because the sentence model is uncased
    text = text.replace('&nbsp;',' ')
    text = text.lower()

    # The maximum token length admitted by is 256
    max_sequence_length = 128
    # Larger texts are cut into 256 token length pieces and their vectors are averaged
    # Take into account that a cell could have more than one token
    if len(text) > max_sequence_length:
        list_texts = []
        for i in range(0, len(text), max_sequence_length):
            list_texts.append(text[i:i+max_sequence_length])
        return list_texts
    else:
        return [text]

def main():
    parser = argparse.ArgumentParser(description='Process WikiTables corpus')
    parser.add_argument('-i', '--input', default='/raid/wake/data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='stb', choices=['stb', 'rbt', 'brt','fst','w2v','blo', 'sci'],
                        help='Model to use: "sbt" (Sentence-BERT, Default), "rbt" (Roberta),"fst" (fastText), "w2v"(Word2Vec), "blo" (Bloomer), "sci" sci-bert '
                             ' "brt" (Bert)')
    parser.add_argument('-r', '--result', default='./result', help='Name of the output folder that stores the similarity values calculated')

    args = parser.parse_args()

    files_path = args.input

    files = os.listdir(files_path)

    invertedIndex = dict()
    if args.model == 'sci' or args.model == 'brt':
        size_vector = 768
    elif args.model == 'stb':
        size_vector = 384
    elif args.model == 'rbt' or args.model=='blo':
        size_vector = 1024
    else:
        size_vector = 300


    #index_abstract = createIndex(size_vector)
    index_headers = createIndex(size_vector)
    index_content = createIndex(size_vector)

    ignored = 0

    for file in tqdm(files):
        if file[-4:] == 'json':
            try:
                with open(files_path+file, 'r') as f:
                    data = json.load(f)
                    key = file.split('.')[0]
                    # Obtenemos el embedding del abstract 'a'
                    #a = proccessText(data['desc'])
                    #a_vec = np.array(getEmbeddings(a), dtype="float32")
                    #if a_vec.shape[0] > 1:
                    #    a_vec =  np.array([np.mean(a_vec, axis=0)], dtype="float32")
                    #faiss.normalize_L2(a_vec)

                    # Headers
                    df = pd.read_csv(files_path+str(data['id'])+'.csv', encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None, nrows=1020)
                    t1 = proccessHeaders(df.columns.values)
                    t1_vec = np.array(getEmbeddings(t1), dtype="float32")
                    if t1_vec.shape[0] > 1:
                        t1_vec = np.array([np.mean(t1_vec, axis=0)], dtype="float32")

                    faiss.normalize_L2(t1_vec)

                    # Content
                    t2 = []
                    if len(df.index)>1000:
                        df = df.sample(frac=0.1, replace=True, random_state=1)
                    
                    for col in df.columns:
                        aux = proccessText(' '.join(df[col].astype(str).tolist()))
                        
                        emb = []
                        if len(aux)>300:
                            for a in range(0,len(aux),300):
                                emb += np.array(getEmbeddings(aux[a:a+300]), dtype="float32")
                        else:
                            emb = np.array(getEmbeddings(aux), dtype="float32")
                      
                        if np.any(emb):
                            if emb.shape[0] > 1:
                                emb =  np.array([np.mean(emb, axis=0)], dtype="float32")
                           
                            faiss.normalize_L2(emb)
                            t2.append(emb)
                    
                    t2_vec = np.array(np.mean(t2, axis=0))

                    id = np.random.randint(0, 99999999999999, size=1)
                
                    invertedIndex[id[0]] = key
                    index_headers.add_with_ids(t1_vec, id)
                    index_content.add_with_ids(t2_vec, id)

            except Exception as e:
                traceback.print_exc()
                #print(t1)
                #print(t2)
                ignored+=1

    saveIndex(index_headers, os.path.join('faiss_data', args.model+'_headers.faiss'))
    saveIndex(index_content, os.path.join('faiss_data', args.model+'_content.faiss'))
    saveInvertedIndex(invertedIndex, os.path.join('faiss_data', args.model+'_invertedIndex'))

    print('Ignored', ignored)

if __name__ == "__main__":
    main()