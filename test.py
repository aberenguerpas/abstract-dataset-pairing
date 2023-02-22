# convert pmid to pmcid 
import ijson, json, os
import pickle
from glob import glob
from tqdm import tqdm

with open('./mapping.pickle', 'rb') as handle:
    b = pickle.load(handle)

all_files = os.listdir('./data2/')


f = open('/raid/wake/raw_data/pubmed_sorted.json')
objects = ijson.items(f, 'articles.item')
for obj in tqdm(objects):
    try:
        aux = dict()
        aux['id'] = b[obj['pmid']]
        aux['abstract'] = obj["abstractText"]
        aux['keywords'] = list(obj["mesh"].values())

        #data = glob("./data2/"+aux["id"]+'*.csv')
        data = [i for i in all_files if aux["id"] in i]

        if data:
            aux['data'] = data
            
            with open("./data3/"+aux['id']+".json", "w") as outfile:
                json.dump(aux, outfile, indent=4)
    except Exception as e:
        pass