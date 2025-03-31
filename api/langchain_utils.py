from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings # Use HuggingFace embeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader
import logging
import os
from typing import List, Dict
import yaml
from api.groq_utils import query_groq #import proper access of groq util method
import re #import preprocess tool

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def preprocess_text(text: str) -> str:
    """Basic text preprocessing (lowercase, remove non-alphanumeric)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


class RealTimeDataProcessor:

    def __init__(self, api_key: str, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                  index_path: str = "data/rag_data/realtime_data_index"):
        self.api_key = api_key
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name) # switch between HF embedding and paid embedding (OpenAi)

        if os.path.exists(self.index_path):
            self.db = FAISS.load_local(self.index_path, self.embeddings)
            logger.info(f"Loaded existing FAISS index from {self.index_path}")
        else:
            self.db = None
            logger.info("No existing FAISS index found. Index will be created on update.")

    def update_index(self, new_data: List[str]):
        """
        Update the FAISS index with new data.

        Args:
            new_data: A list of strings representing the new data to add to the index.
        """
        try:
            #first preprocess this code it can create some un neat info inside Vector Store before putting
            processed_data = [preprocess_text(text) for text in new_data]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=config['rag']['chunk_size'], chunk_overlap=config['rag']['chunk_overlap']) #load those parameter from configuration
            chunks = text_splitter.create_documents(processed_data)

            if self.db is None:
                self.db = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.db.add_documents(chunks)

            self.db.save_local(self.index_path)
            logger.info(f"Updated FAISS index saved to {self.index_path}")

        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}", exc_info=True)
            raise

    async def retrieve_context(self, query: str, top_k: int = 5) -> List[str]: #default top k five for more flexibility

        if self.db is None:
            logger.warning("No FAISS index available.  Returning an empty context.")
            return []

        try:
            docs = self.db.similarity_search(query, k=top_k)
            context = [doc.page_content for doc in docs]
            logger.info(f"Retrieved context: {context}")
            return context
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}", exc_info=True)
            return []

    async def query_rag(self, query: str, api_key: str, model: str) -> Dict[str, any]:
        """
        Query the RAG system with a given query.

        Args:
            query (str): The query string.

        Returns:
            Dict[str, any]: A dictionary containing the response, including the source documents used for context.
        """
        try:
            context = await self.retrieve_context(query)
            if not context:
                return {"answer": "I'm sorry, I don't have information to answer this query.","source_documents": []}

            context_str = "\n".join(context)
            prompt = f"""Use the following context to answer the question at the end. If you can't find the answer, just say that you don't know, don't try to make up an answer.
            Context: {context_str}
            Question: {query}"""

            response = await query_groq(prompt, api_key, model, 0.1)
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"answer": answer, "source_documents": context}

        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            return {"answer": f"Error during RAG query: {e}", "source_documents": []}

# Example usage (can be placed in a separate utility function file):
async def load_data_from_web(url: str) -> str:

    try:
        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config['rag']['chunk_size'], chunk_overlap=config['rag']['chunk_overlap']) # load parameters from conf
        chunks = text_splitter.split_documents(data)

        all_content = [chunk.page_content for chunk in chunks]

        # Combine the content into a single string
        combined_content = '\n'.join(all_content)

        logger.info(f"Data loaded from {url} successfully")
        return combined_content

    except Exception as e:
        logger.error(f"Error loading data from {url}: {e}", exc_info=True)
        return ""
