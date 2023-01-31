import time, os, json
import argparse
from utils import *
from tqdm import tqdm
import requests
import faiss
import numpy as np
import traceback

def getEmbeddings(data):
    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})

    if response.status_code == 200:
        return response.json()['emb']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod)


def proccessText(text):
    # Lowercase the text because the sentence model is uncased
    text = text.replace('&nbsp;',' ')
    text = text.lower()

    # The maximum token length admitted by is 256
    max_sequence_length = 256
    # Larger texts are cut into 256 token length pieces and their vectors are averaged
    # Take into account that a cell could have more than one token
    if len(text) > max_sequence_length:
        list_texts = []
        for i in range(0, len(text), max_sequence_length):
            list_texts.append(text[i:i+max_sequence_length])
        return list_texts
    else:
        return [text]


def search(vec_abstract, index_h, index_c, inverted, alpha):

    ids_list = []

    faiss.normalize_L2(vec_abstract)

    # Headers search
    distances_h, indices_h = index_h.search(vec_abstract, index_h.ntotal - 1)
    results_h = [(inverted[r], distances_h[0][i]) for i, r in enumerate(indices_h[0])]
    ids_list += [k for k,_ in results_h]

    # Content search
    distances_c, indices_c = index_c.search(vec_abstract, index_c.ntotal - 1)
    results_c = [(inverted[r], distances_c[0][i]) for i, r in enumerate(indices_c[0])]
    ids_list += [k for k,_ in results_c]

    ids_list = list(set(ids_list)) # List with candidate tables id, delte duplicates

    # Ranking documents
    ranking = dict()

    for id in ids_list:
        ranking[id]= getScore(id, results_h, results_c, alpha)
        
    # Ordenar ranking
    ranking_sort = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
   
    return list(map(lambda x: x[0], ranking_sort[:10])) # Nos quedamos con el top 10
    #return ranking_sort[:5]

def getScore(id, results_h, results_c, alpha):

    #Filter only results from an id
    score_c = list(filter(lambda d: d[0]==id, results_c))
    score_h = list(filter(lambda d: d[0]==id, results_h))

    if len(score_h)>0:
        score_h = score_h[0][1]
    else:
        score_h = 0

    if len(score_c)>0:
        score_c = score_c[0][1]
    else:
        score_c = 0

    score = score_h*alpha + score_c*(1-alpha)

    return score

def checkPos(id,lis):
    count = 1
    for item in reversed(lis):
        if item == id:
            return count/len(lis)
        count+=1
    return 0

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Classify abstracts')
    parser.add_argument('-i', '--input', default='data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-n', '--indexdir', default='faiss_data/', help='Index Dir')
    parser.add_argument('-m', '--model', default='w2v', choices=['stb', 'rbt','brt','fst','w2v','sci','blo'],
                        help='"stb" (Sentence-BERT), "rbt" (Roberta),"fst" (FastText),"w2v" (Word2Vec) or "brt" (BERT)')
    parser.add_argument('-c', '--clasification', default='clasification/', help='Name of the output folder that stores the classification results')
    
    args = parser.parse_args()

    # Headers collection
    index_headers = loadIndex(os.path.join(args.indexdir, args.model+'_headers.faiss'))
    # Content collection
    index_content = loadIndex(os.path.join(args.indexdir, args.model+'_content.faiss'))
    # Read inversed Index
    inverted = loadInversedIndex(os.path.join(args.indexdir, args.model+'_invertedIndex'))

    # Counters
    mmr = dict()
    for alpha in np.arange(0, 1.1, 0.1):
        mmr[alpha] = []
    
    # Read abstracts
    files = os.listdir(args.input)
    files = [i for i in files if i.endswith(".json")]
    for file in tqdm(files):

        #load table
        if file[-4:] == 'json':
            try:
                with open(os.path.join(args.input,file), 'r') as f:
                    data = json.load(f)
                   
                    if len(data['desc'].replace('&nbsp;',' ').split(" "))>140:
                        abstract = proccessText(data['desc'])
                        # Create embedding abstract
                        vec_abstract = np.array(getEmbeddings(abstract)).astype(np.float32)

                        # Search
                        for alpha in np.arange(0, 1.1, 0.1):
                            rank = search(vec_abstract, index_headers, index_content, inverted, alpha)
                            # Check in results are correct
                            points = checkPos(file[:-5], rank)        
                            mmr[alpha].append(points)

            except Exception as e:
                print(e)
                traceback.print_exc()

    # Create a file with the results
    results = ["Results model "+args.model+"\n"]

    for alpha in np.arange(0, 1.1, 0.1):
        results.append('alpha ' + str(round(alpha,1))+ ' MMR: '+ str(round(np.mean(mmr[alpha]),2)) + "\n")
   
    print('Search time: ' + str(round(time.time() - start_time, 2)) + ' seconds')

    f = open(args.clasification+args.model+".txt", "w")
    f.writelines(results)
    f.close()
if __name__ == "__main__":
    main()