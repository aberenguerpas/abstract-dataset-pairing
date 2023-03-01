import os, json, sys, csv
import faiss
import numpy as np
from tqdm import tqdm
import requests
from utils import *
import pandas as pd
import time
import argparse
import logging
from glob import glob


def getEmbeddings(data):
    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})
    if response.status_code == 200:
        return response.json()['emb']
    else:
        logger.warning(msg="Problem getting embeddings" + str(response.status_code))


def proccessHeaders(headers):
    # Lowercase the text because the sentence model is uncased
    headers = [ h if not 'Unnamed:' in h else '' for h in headers]
    headers = ' '.join(headers)
    headers = headers.replace('.',' ')
    headers = headers.replace('_',' ')
    headers = headers.replace('-',' ')
    #headers = re.sub(r'(?<!^)(?=[A-Z])', ' ', headers)
    headers = headers.lower()
    #headers = headers.replace('&nbsp;',' ')
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

# Check % of numeric column
def isNumCol(df):
    num_cols = 0
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            num_cols+=1
    return num_cols/len(df.columns)

# Check % of numeric column names
def isNum(df):
    num_cols = 0
    for col in df.columns:
        col = col.replace('-','')
        col = col.replace('.','')
        if col.isnumeric():
            num_cols+=1
    return num_cols/len(df.columns)

def main():
    csv.field_size_limit(sys.maxsize)
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process WikiTables corpus')
    parser.add_argument('-i', '--input', default='/raid/wake/data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='stb', choices=['stb', 'rbt', 'brt','fst','w2v','blo', 'sci'],
                        help='Model to use: "sbt" (Sentence-BERT, Default), "rbt" (Roberta),"fst" (fastText), "w2v"(Word2Vec), "blo" (Bloomer), "sci" sci-bert '
                             ' "brt" (Bert)')
    parser.add_argument('-r', '--result', default='./result', help='Name of the output folder that stores the similarity values calculated')

    args = parser.parse_args()

    files_path = args.input

    files_path ="./data3/"
    files_path2 ="./data2/"

    files = [i for i in os.listdir(files_path) if i.endswith(".json")]
  
    invertedIndex = dict()
    if args.model == 'sci' or args.model == 'brt':
        size_vector = 768
    elif args.model == 'stb':
        size_vector = 384
    elif args.model == 'rbt' or args.model=='blo':
        size_vector = 1024
    else:
        size_vector = 300

    index_headers = createIndex(size_vector)
    index_content = createIndex(size_vector)
    index_keywords = createIndex(size_vector)

    ignored = 0
    discard = []
    for file in tqdm(files):
        try:
            key = ""
            with open(os.path.join(files_path, file), 'r') as f:
                meta = json.load(f)
                key = meta['id']


            for dfile in meta['data']:
                if not dfile.endswith(".csv"):
                    continue
                dfile = os.path.join(files_path2, dfile)
                
                df=""
                #file = os.path.join(files_path, file)
                # Headers
                try:
                    df = pd.read_csv(dfile , encoding = "utf-8-sig", on_bad_lines='skip', engine='python', sep = None, nrows=1020)
                except Exception:
                    df = pd.read_csv(dfile , encoding = "unicode_escape", on_bad_lines='skip', engine='python', sep = None, nrows=1020)
                
                # Check if headers are numeric or if all columns are numeric
                #print(df.head())
                if isNum(df)>0.2 and len(df.columns.values)<2: #or isNumCol(df)==1:
                    ignored+=1
                    discard.append(key)
                    continue
                
                # Add keywords
                t0 = [" ".join(meta['keywords'])]
                
                t0_vec = np.array([getEmbeddings(t0)], dtype="float32")
  
                faiss.normalize_L2(t0_vec)

                if len(t0_vec.shape)>2:
                    t0_vec = t0_vec[0]

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
                    #if not df[col].dtype.kind in 'biufc': # Elimina las numericas
                    aux = proccessText(' '.join(df[col].astype(str).tolist()))
                    
                    emb = ""
                    # Split text - too big , memory problems
                    if len(aux) > 300:
                        for a in range(0, len(aux), 300):
                            if a==0:
                                emb = getEmbeddings(aux[a:a+300])
                            else:
                                emb = np.append(emb, getEmbeddings(aux[a:a+300]), axis=0)
                    else:
                        emb = getEmbeddings(aux)

                    emb = np.array(emb, dtype="float32")
                
                    if np.any(emb):
                        if emb.shape[0] > 1:
                            emb =  np.array([np.mean(emb, axis=0)], dtype="float32")
                    
                        faiss.normalize_L2(emb)
                        t2.append(emb)
                
                if len(t2) > 0:
                    t2_vec = np.array(np.mean(t2, axis=0))

                    id = np.random.randint(0, 99999999999999, size=1)
                    invertedIndex[id[0]] = key
                    index_headers.add_with_ids(t1_vec, id)
                    index_content.add_with_ids(t2_vec, id)
                    index_keywords.add_with_ids(t0_vec, id)
                else:
                    discard.append(key)

        except Exception as e:
            print('No vector available')
            logger.error("Exception occurred ", exc_info=True)
            logger.error("Problem with file "+ file)
            ignored += 1

    logger.info("Saving indexes "+ args.model + "...")
    saveIndex(index_headers, os.path.join('faiss_data', args.model+'_headers.faiss'))
    saveIndex(index_content, os.path.join('faiss_data', args.model+'_content.faiss'))
    saveIndex(index_keywords, os.path.join('faiss_data', args.model+'_keywords.faiss'))
    saveInvertedIndex(invertedIndex, os.path.join('faiss_data', args.model+'_invertedIndex'))
    # Save discarted abstracts
    saveInvertedIndex(discard, './disc')

    logger.info("Files ignored "+str(ignored))
    logger.info('Indexation time '+ args.model+ ': ' + str(round(time.time() - start_time, 2)) + ' seconds')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='index_logger')

    logFileFormatter = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s Function %(funcName)s Line %(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(filename='execute.log')
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(level=logging.INFO)
    logger.addHandler(fileHandler)
    main()