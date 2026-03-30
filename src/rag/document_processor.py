import os
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        """
        Initializes the processor with strict Markdown header rules.
        """
        #We tell the splitter to look to H1, H2, H3 tags
        self.headers_to_split_on = [
            ("#", "Section"),
            ("##", "Subsection"),
            ("###", "Topic")
        ]

        # strip_headers=False ensures the LLM still gets to read the title of the section
        self.markdown_splitter = MarkdownHeaderTextSplitter(
          headers_to_split_on=self.headers_to_split_on,
          strip_headers=False  
        )
    
    def load_and_chunk_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Reads a markdown file and splits it into logical, metadata-rich chunks.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Critical Error: Policy file not found at: {file_path}")
        

        print(f"Loading document from: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            markdown_documents = f.read()
        # Perform the intelligent split
        splits = self.markdown_splitter.split_text(markdown_documents)
        # Reformat into a clean, production-friendly dictionary format
        processed_chunks = []
        for split in splits:
            processed_chunks.append({
                "content": split.page_content,
                "metadata": split.metadata
            })

        logger.info(f"Successfully split document into {len(processed_chunks)} logical chunks")
        return processed_chunks
    

if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent


    policy_path = project_root / "docs"  / "ecommerce_policies.md"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s",)
    logger.info("Policy base: %s", policy_path)

    processor = DocumentProcessor()

    chunks = processor.load_and_chunk_markdown(policy_path)


    logger.info("Created %d chunks", len(chunks))
    #Print the first real chunk to verify it worked
    print("\n-- SAMPLE CHUNK - ")
    if chunks:
        #print(f"METADATA: {chunks[1]["metadata"]}")
        #print(f"CONTENT:\n{chunks[1]["content"][:150]}...")
        print(chunks[1])

      