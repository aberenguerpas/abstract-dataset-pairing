import requests
import torch 
import numpy as np

def getEmbeddings(data):
    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})
    if response.status_code == 200:
        return response.json()['emb']
    else:
        return []


a = ['DNA is a key element in biology']

b = a[0].split(" ")
print(b)
c = ['lie like a horse 876']

a = np.array(getEmbeddings(a))[0]
b = np.mean(getEmbeddings(b), axis=0)
c = np.array(getEmbeddings(c))[0]

cos = torch.nn.CosineSimilarity(dim=0)
sim = cos(torch.from_numpy(a), torch.from_numpy(b))
sim2 = cos(torch.from_numpy(a), torch.from_numpy(c))

print('Similitud:', float(sim))
print('no Similitud:', float(sim2))