import time, os, json
import argparse
from utils import *
from tqdm import tqdm
import requests
import faiss
import numpy as np
import logging

def getEmbeddings(data):
    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})

    if response.status_code == 200:
        return response.json()['emb']
    else:
        logger.warning(msg="Problem getting embeddings" + response.status_code)


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
   
    return list(map(lambda x: x[0], ranking_sort[:5])) # Nos quedamos con el top 10
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

def checkPos(id, lis):
    count = 1
    for item in reversed(lis):
        if item == id:
            return count/len(lis)
        count+=1
    return 0

def checkPrecision(id, lis):
    res = []

    for i in [1,3,5]:
        if id in lis[:i]:
            res.append(1)
        else:
            res.append(0)

    return res


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

     # Read discarted
    discarted = loadInversedIndex('./disc')

    # Counters
    mmr = dict()
    precision = dict()
    for alpha in np.arange(0, 1.1, 0.1):
        mmr[alpha] = []
        precision[alpha] = []
    
    # Read abstracts
    files = os.listdir(args.input)
    files = [i for i in files if i.endswith(".json")]

    try:
        for i in discarted:
            files.remove(i+".json")
    except Exception as e:
            logger.error("Exception occurred", exc_info=True)

    files.sort()
    logger.info("Classifying  " + str(len(files))+ " files...")
    for file in tqdm(files):
        try:
            with open(os.path.join(args.input,file), 'r') as f:
                data = json.load(f)
                
                abstract = proccessText(data['desc'])
                # Create embedding abstract
                vec_abstract = np.array(getEmbeddings(abstract)).astype(np.float32)

                # Search
                for alpha in np.arange(0, 1.1, 0.1):
                    rank = search(vec_abstract, index_headers, index_content, inverted, alpha)
                    # Check in results are correct
                    
                    precision[alpha].append(checkPrecision(file[:-5], rank))
                    points = checkPos(file[:-5], rank)        
                    mmr[alpha].append(points)

        except Exception as e:
            logger.error("Exception occurred", exc_info=True)
            #traceback.print_exc()

    # Create a file with the results
    results = ["Results model "+args.model+"\n"]
    logger.info("Saving info from model "+args.model)
    results.append("-------------------------------------- \n")

    for alpha in np.arange(0, 1.1, 0.1):
        text = 'alpha ' + str(round(alpha,1))+ '- MMR: '+ str(round(np.mean(mmr[alpha]),2))
        logger.info(text)
        results.append(text + "\n")

    results.append("-------------------------------------- \n")
    for alpha in np.arange(0, 1.1, 0.1):
        text = 'alpha ' + str(round(alpha,1))+ '. Precision Top 1/3/5: '+ str(np.sum(precision[alpha], axis=0)/len(precision[alpha]))
        logger.info(text)
        results.append(text + "\n")

    

    logger.info('Search time '+ args.model+ ': ' + str(round(time.time() - start_time, 2)) + ' seconds')

    f = open(args.clasification+args.model+".txt", "w")
    f.writelines(results)
    f.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='classifier_logger')

    logFileFormatter = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s Function %(funcName)s Line %(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(filename='execute.log')
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(level=logging.INFO)
    logger.addHandler(fileHandler)
    
    main()