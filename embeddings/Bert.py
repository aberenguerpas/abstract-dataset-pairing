from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
class Bert:
    """
    def __init__(self):
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dimensions = 768

    def getEmbedding(self, data):
        torch.cuda.empty_cache()
        with torch.no_grad():
            tab = self.tokenizer(
                    data,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
            ).to(self.device)

            self.model = self.model.to(self.device)
            output = self.model(**tab)
            return [i[0] for i in output.last_hidden_state]


    """
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens') #device='cuda'
        self.tokenizer = None
        self.dimensions = 768

    def getEmbedding(self, data):
        return self.model.encode(data)
