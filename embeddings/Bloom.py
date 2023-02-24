from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class Bloom:

    def __init__(self):
        torch.cuda.empty_cache()
        self.model = AutoModel.from_pretrained("bigscience/bloom-560m", output_hidden_states = True)
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dimensions = 768

    def getEmbedding(self, data):
        torch.cuda.empty_cache()

        with torch.no_grad():
            tab = self.tokenizer(
                    data,
                    padding=True,
                    return_tensors="pt"
            ).to(self.device)

            self.model = self.model.to(self.device)
            output = self.model(**tab)
            return [i[0] for i in output.last_hidden_state]

