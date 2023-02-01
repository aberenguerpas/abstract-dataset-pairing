import requests
import torch 
import numpy as np

def getEmbeddings(data):
    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':data})
    if response.status_code == 200:
        return response.json()['emb']
    else:
        return []


a = ['I have an apple in my fridge']

b = a[0].split(" ")

c = ['boat']

a = np.array(getEmbeddings(a))[0]
b = np.mean(getEmbeddings(b), axis=0)
c = np.array(getEmbeddings(c))[0]

cos = torch.nn.CosineSimilarity(dim=0)
sim = cos(torch.from_numpy(a), torch.from_numpy(b))
sim2 = cos(torch.from_numpy(a), torch.from_numpy(c))

print('Similitud:', float(sim))
print('no Similitud:', float(sim2))