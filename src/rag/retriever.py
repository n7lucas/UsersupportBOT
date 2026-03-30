import numpy as np
import logging
from src.rag.embedding_model import EmbeddingModel
from src.rag.vector_store import PolicyVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class SemanticRetriever:

    def __init__(self, vector_store_path="policy_vector_store.pkl"):
        """
        Loads the embedder and the vector store database into memory.
        """

        self.embedder = EmbeddingModel()
        self.vector_db = PolicyVectorStore(vector_store_path)

        self.knowledge_base = self.vector_db.load_store()

    def search(self, user_query: str, top_k : int = 2):
        """
        Embeds the query and mathematically finds the most relevant policy chunks.
        """

        logger.info(f"Executing semantic search for: '{user_query}' ")

        query_emb = np.array(self.embedder.encode([user_query])[0], dtype=np.float32)

        results = []

        #Compare query tih chunks in db
        for chunk in self.knowledge_base:
            chunk_embedding = chunk["embedding"]

            # --- THE MAGIC MATH: COSINE SIMILARITY ---
            # Dot Product divided by the product of the magnitudes (norms)
            dot_product = np.dot(query_emb, chunk_embedding)
            norm_query = np.linalg.norm(query_emb)
            norm_chunk = np.linalg.norm(chunk_embedding)

            similarity_score = dot_product / (norm_query * norm_chunk)

            results.append({
                "score": similarity_score,
                "metadata": chunk["metadata"],
                "content": chunk["content"]
            })


        #Sort results by similarity
        results.sort(key= lambda x: x["score"], reverse=True)

        #Return top k
        top_results = results[:top_k]

        logger.info(f"Found top {top_k} matches. Highest score: {top_results[0]['score']:.4f}")
        return top_results
    
if __name__ == "__main__":

    retriever = SemanticRetriever("policy_vector_store.pkl")

    test_question = "How many days do I have to return an item if it's broken?"

    print(f"\nUser Question: {test_question}")

    matches = retriever.search(test_question, top_k=3)

    print("\n--- TOP MATCHES ---")
    for i, match in enumerate(matches, 1):
        print(f"\nMatch #{i} | Score: {match['score']:.4f} | Section: {match['metadata']}")
        # Print just the first 150 characters to keep the console clean
        print(f"Preview: {match['content'][:150]}...")