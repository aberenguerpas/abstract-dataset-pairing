import os, json
import faiss
import numpy as np
from tqdm import tqdm
import requests
from utils import *
import pandas as pd
import traceback


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

def main():
    files_path = './data/'
    files = os.listdir(files_path)

    invertedIndex = dict()

    size_vector = 384

    index_abstract = createIndex(size_vector)
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
                    a = proccessText(data['desc'])

                    a_vec = np.array([getEmbeddings(a)], dtype="float32")

                    if len(a_vec[0])>1:
                        a_vec = np.array([[np.mean(a_vec[0], axis=0)]])
                    faiss.normalize_L2(a_vec)

                    # headers
                    df = pd.read_csv(files_path+str(data['id'])+'.csv', encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None, nrows=500)
                    t1 = proccessHeaders(df.columns.values)
                    t1_vec = np.array(getEmbeddings(t1), dtype="float32")
                    if len(t1_vec)>1:
                        t1_vec = np.array([np.mean(t1_vec, axis=0)])
                        #print(t1_vec.shape)
                  

                    faiss.normalize_L2(t1_vec)

                    # Content
                    t2 = []
                    
                    if len(df.index)>1000:
                        df = df.sample(frac=0.1, replace=True, random_state=1)
                    
                    for col in df.columns:
                        aux = proccessText(' '.join(df[col].astype(str).tolist()))
                        emb = getEmbeddings(aux)
                        if emb != []:
                            aux = np.mean(emb, axis=0)
                            t2.append(aux)
                    
                    t2_vec = np.array([np.mean(t2, axis=0)])

                    id = np.random.randint(0, 99999999999999, size=1)
                
                    index_abstract.add_with_ids(a_vec[0], id)
                    index_headers.add_with_ids(t1_vec, id)
                    index_content.add_with_ids(t2_vec, id)

                    invertedIndex[id[0]] = key
            except Exception as e:
                print(t1_vec.shape)
                traceback.print_exc()
                ignored+=1
                continue
        
    saveIndex(index_abstract, os.path.join('faiss_data', 'model_abstract.faiss'))
    saveIndex(index_headers, os.path.join('faiss_data', 'model_headers.faiss'))
    saveIndex(index_content, os.path.join('faiss_data', 'model_content.faiss'))

    print('Ignored', ignored)

if __name__ == "__main__":
    main()