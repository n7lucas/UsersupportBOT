from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode (self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def encode_dataframe(self, df, column="instruction"):

        #fillna to prevent any acidental Nan value and trnaforme tolist bc transformer expect a list
        sentence = df[column].fillna("").tolist()

        embeddings = self.encode(sentence)

        return embeddings
    
    def save_embeddings(self, embeddings, path):
        np.save(path, embeddings)
        
        
