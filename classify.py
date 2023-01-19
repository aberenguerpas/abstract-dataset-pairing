import time, os
import argparse
from utils import *
from tqdm import tqdm
import pandas as pd

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Search in wikitables')
    parser.add_argument('-i', '--input', default='data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-n', '--indexDir', default='faiss_data')
    parser.add_argument('-m', '--model', default='brt', choices=['stb', 'rbt','brt','fst','w2v','sci','blo'],
                        help='"stb" (Sentence-BERT), "rbt" (Roberta),"fst" (FastText),"w2v" (Word2Vec) or "brt" (BERT)')
    parser.add_argument('-c', '--clasification', default='clasification', help='Name of the output folder that stores the classification results')


    args = parser.parse_args()

    # Headers collection
    index_headers = loadIndex(os.path.join(args.indexDir, args.model+'_headers.faiss'))

    # Content collection
    index_content = loadIndex(os.path.join(args.indexDir, args.model+'_content.faiss'))

    # Read inversed Index
    inverted = loadInversedIndex(os.path.join(args.indexDir, args.model+'_invertedIndex'))

    # Read abstracts
    for path in tqdm(os.listdir(args.input)):
        #load table
        file = open(os.path.join(args.data, path)+'.csv')
        id = queries[queries[1]==path].iloc[:,0].values[0]
        table = pd.read_csv(file)

        # create embeddings
        embs = create_embeddings(table)

        # search
        rank = search(embs, index_headers, index_content, inverted, path)

        # save result
        save_result(id, rank, args.result+'_'+args.model+'.csv')
    
    print('Search time: ' + str(round(time.time() - start_time, 2)) + ' seconds')



if __name__ == "__main__":
    main()