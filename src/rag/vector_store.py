import joblib
import numpy as np
import os
from src.rag.document_processor import DocumentProcessor
from src.rag.embedding_model import EmbeddingModel
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class PolicyVectorStore:

    def __init__(self, store_path="policy_vector_store.pkl"):
        """
        Manages the embedding and storage of policy chunks.
        """

        self.store_path = store_path
        self.embedder = EmbeddingModel()
        self.processor = DocumentProcessor()
        self.knowledge_base = []


    def build_store(self, markdown_file_path: str):

        """
        
        """
        #Load chunk data
        logger.info("1 - Processing Markdown Document...")
        chunks =  self.processor.load_and_chunk_markdown(markdown_file_path)

        ""
        logger.info(f"Generating embedding vector for {len(chunks)} chunks")
        text_to_embed = [chunk["content"] for chunk in chunks]

        embeddings = self.embedder.encode(text_to_embed)

        logger.info("3. Assemblign the Knowledge Base")

        self.knowledge_base = []
        for i, chunk in enumerate(chunks):
            self.knowledge_base.append({
                "metadata": chunk["metadata"],
                "content": chunk["content"],
                "embedding": np.array(embeddings[i], dtype=np.float32)
            })

       #Save to hard drive in order to the Planner load it instantly
       
        logger.info(f"4. Saving Vector Store to {self.store_path}...")

        joblib.dump(self.knowledge_base, self.store_path)

        print("Success! Vector Store is build and saved")

    def load_store(self):  

        if not os.path.exists(self.store_path):
            raise FileNotFoundError(f"Vector store not found at {self.store_path}")    
        
        self.knowledge_base = joblib.load(self.store_path)
        print(f"Vector Store loaded. ({len(self.knowledge_base)} policy chunks ready)")

        return self.knowledge_base
    

if __name__ == "__main__":
    
    #Load markdown file
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    markdown_path = project_root / "docs"  / "ecommerce_policies.md"

    #Initialize and build the database
    vector_store = PolicyVectorStore("policy_vector_store.pkl")
    vector_store.build_store(markdown_path)

    #Verify it loads corectly
    vector_db = vector_store.load_store()
    logger.info( f"First chunk meadata: {vector_db[0]["metadata"]}")
    logger.info(f"First chunk embedding shape: {vector_db[0]["embedding"].shape}")

