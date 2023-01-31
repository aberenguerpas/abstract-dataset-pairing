import argparse
import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from SentenceBert import SentenceBert
from Bert import Bert
from Roberta import Roberta
from FastText import FastText
from W2vec import Word2Vec
from Bloom import Bloom
from SciBert import SciBert
import logging

app = FastAPI()

def checkGPU():
    if torch.cuda.is_available():
        print("Cuda GPU available")
        print(torch.cuda.device_count(), "Devices")
        print("Current device:")
        print(torch.cuda.get_device_name(0))

    elif torch.backends.mps.is_available():
        print("Mac M1 acceleration available!")


@app.post("/getEmbeddings")
async def getEmbeddings(request: Request):
    response = await request.json()
    data = model.getEmbedding(response['data'])
    data = [l.tolist() for l in data]
    return JSONResponse(content={'emb':data})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embeddings microservice')
    parser.add_argument('-m', '--model', default='stb', choices=['stb','brt','rbt','fst','w2v', 'blo', 'sci'],
                        help='Model to use: "stb" (Sentence-Bert, by default), "brt" (bert-base-uncased),'
                             ' "rbt" (Roberta), "fst" (fastText), "w2v" (Word2Vec), "blo" (Bloom), "sci" (sci-bert)')
    args = parser.parse_args()

    checkGPU()

    if args.model == 'stb':
        model = SentenceBert()
    elif args.model == 'brt':
        model = Bert()
    elif args.model == 'blo':
        model = Bloom()
    elif args.model == 'sci':
        model = SciBert()
    elif args.model == 'rbt':
        model = Roberta()
    elif args.model == 'fst':
        model = FastText()
    elif args.model == 'w2v':
        model = Word2Vec()
    else:
        model = SentenceBert()

    uvicorn.run(app, host="0.0.0.0", port=5000)
