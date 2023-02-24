# convert pmid to pmcid 
import ijson, json, os
import pickle
from glob import glob
from tqdm import tqdm

with open('./mapping.pickle', 'rb') as handle:
    b = pickle.load(handle)

all_files = os.listdir('./data2/')

count = 0
f = open('/Users/albertoberenguerpastor/Downloads/pubmed_sorted.json')
objects = ijson.items(f, 'articles.item')
for obj in tqdm(objects):
    try:
        aux = dict()
        aux['id'] = b[obj['pmid']]
        aux['abstract'] = obj["abstractText"]
        aux['keywords'] = list(obj["mesh"].values())
        if int(obj['year'])>2016:
            data = [i for i in all_files if aux["id"] in i]

            if data:
                aux['data'] = data
                print(count)
                count+=1
          
                with open("./data3/"+aux['id']+".json", "w") as outfile:
                        json.dump(aux, outfile, indent=4)
                if count>20000:
                    break
    except Exception as e:
        #print(e)
        pass

print("Obtained", count)