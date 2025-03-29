from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Retriever(ABC):
    """
    Abstract base class for retrievers.
    
    This interface allows the ReaRAG implementation to work with different retrieval
    sources (such as web search, vector databases, or knowledge graphs) through a
    common API.
    """
    
    @abstractmethod
    def retrieve(self, query: str) -> str:
        """
        Retrieve information based on the given query.
        
        Args:
            query: The search query
            
        Returns:
            The retrieved information as a string
        """
        pass


class WebSearchRetriever(Retriever):
    """
    Retriever implementation for web search.
    """
    
    def __init__(self, search_provider: str = "google", api_key: Optional[str] = None, **kwargs):
        """
        Initialize the web search retriever.
        
        Args:
            search_provider: The search provider to use (e.g., "google", "bing")
            api_key: API key for the search provider
            **kwargs: Additional arguments to pass to the search provider
        """
        self.search_provider = search_provider
        self.api_key = api_key
        self.kwargs = kwargs
        
        if search_provider.lower() == "google":
            try:
                from googleapiclient.discovery import build
            except ImportError:
                raise ImportError("Please install google-api-python-client to use Google search: pip install google-api-python-client")
            
            self.service = build("customsearch", "v1", developerKey=api_key)
        elif search_provider.lower() == "bing":
            # Initialize Bing Search API client
            self.headers = {
                "Ocp-Apim-Subscription-Key": api_key
            }
            self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        else:
            raise ValueError(f"Unsupported search provider: {search_provider}")
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve information using web search.
        
        Args:
            query: The search query
            
        Returns:
            The retrieved information as a string
        """
        try:
            if self.search_provider.lower() == "google":
                # Perform Google search
                result = self.service.cse().list(
                    q=query,
                    cx=self.kwargs.get("cx", ""),  # Search engine ID
                    num=self.kwargs.get("num", 5)
                ).execute()
                
                # Format the results
                if "items" in result:
                    formatted_result = ""
                    for item in result["items"]:
                        formatted_result += f"{item['title']}\n"
                        formatted_result += f"{item['link']}\n"
                        formatted_result += f"{item.get('snippet', '')}\n\n"
                    return formatted_result
                else:
                    return "No results found."
                
            elif self.search_provider.lower() == "bing":
                # Perform Bing search
                import requests
                
                params = {
                    "q": query,
                    "count": self.kwargs.get("count", 5),
                    "offset": self.kwargs.get("offset", 0),
                    "mkt": self.kwargs.get("mkt", "en-US")
                }
                
                response = requests.get(self.search_url, headers=self.headers, params=params)
                response.raise_for_status()
                result = response.json()
                
                # Format the results
                if "webPages" in result and "value" in result["webPages"]:
                    formatted_result = ""
                    for item in result["webPages"]["value"]:
                        formatted_result += f"{item['name']}\n"
                        formatted_result += f"{item['url']}\n"
                        formatted_result += f"{item.get('snippet', '')}\n\n"
                    return formatted_result
                else:
                    return "No results found."
            
            else:
                return "Unsupported search provider."
        
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return f"Error during search: {str(e)}"


class VectorDBRetriever(Retriever):
    """
    Retriever implementation for vector databases.
    """
    
    def __init__(
        self, 
        db_type: str = "chromadb", 
        collection_name: str = "default", 
        embedding_model: Optional[str] = None,
        top_k: int = 3,
        **kwargs
    ):
        """
        Initialize the vector database retriever.
        
        Args:
            db_type: The type of vector database to use
            collection_name: The name of the collection to query
            embedding_model: The embedding model to use for query embedding
            top_k: Number of results to return
            **kwargs: Additional arguments specific to the vector database
        """
        self.db_type = db_type
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.kwargs = kwargs
        
        if db_type.lower() == "chromadb":
            try:
                import chromadb
            except ImportError:
                raise ImportError("Please install chromadb to use ChromaDB: pip install chromadb")
            
            # Initialize ChromaDB client
            self.client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except Exception:
                logger.info(f"Creating new collection: {collection_name}")
                self.collection = self.client.create_collection(name=collection_name)
        
        elif db_type.lower() == "pinecone":
            try:
                import pinecone
            except ImportError:
                raise ImportError("Please install pinecone-client to use Pinecone: pip install pinecone-client")
            
            # Initialize Pinecone
            pinecone.init(
                api_key=kwargs.get("api_key", ""),
                environment=kwargs.get("environment", "")
            )
            
            # Get or create index
            self.index_name = collection_name
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating new index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=kwargs.get("dimension", 768),
                    metric=kwargs.get("metric", "cosine")
                )
            
            self.index = pinecone.Index(self.index_name)
        
        else:
            raise ValueError(f"Unsupported vector database: {db_type}")
        
        # Initialize embedding function if provided
        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer(embedding_model)
            except ImportError:
                raise ImportError("Please install sentence-transformers to use custom embedding model: pip install sentence-transformers")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using the specified embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        if hasattr(self, "embed_model"):
            return self.embed_model.encode(text).tolist()
        else:
            # Use default embedding method for the database
            if self.db_type.lower() == "chromadb":
                # ChromaDB has built-in embedding
                return None
            else:
                raise ValueError("No embedding model specified and database requires explicit embedding")
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve information from the vector database.
        
        Args:
            query: The query to search for
            
        Returns:
            The retrieved information as a string
        """
        try:
            if self.db_type.lower() == "chromadb":
                # Get query embedding if we have a custom model
                query_embedding = self._get_embedding(query) if hasattr(self, "embed_model") else None
                
                # Query the collection
                results = self.collection.query(
                    query_texts=[query] if query_embedding is None else None,
                    query_embeddings=[query_embedding] if query_embedding is not None else None,
                    n_results=self.top_k
                )
                
                # Format the results
                if results and "documents" in results and results["documents"]:
                    formatted_result = ""
                    for doc in results["documents"][0]:
                        formatted_result += f"{doc}\n\n"
                    return formatted_result
                else:
                    return "No relevant documents found."
            
            elif self.db_type.lower() == "pinecone":
                # Get query embedding
                query_embedding = self._get_embedding(query)
                
                # Query the index
                results = self.index.query(
                    vector=query_embedding,
                    top_k=self.top_k,
                    include_metadata=True
                )
                
                # Format the results
                if results and "matches" in results:
                    formatted_result = ""
                    for match in results["matches"]:
                        if "metadata" in match and "text" in match["metadata"]:
                            formatted_result += f"{match['metadata']['text']}\n\n"
                    return formatted_result
                else:
                    return "No relevant documents found."
            
            else:
                return "Unsupported vector database."
        
        except Exception as e:
            logger.error(f"Error during vector DB retrieval: {e}")
            return f"Error during retrieval: {str(e)}"


class SimpleRetriever(Retriever):
    """
    A simple in-memory retriever for testing and demo purposes.
    """
    
    def __init__(self, knowledge_base: Dict[str, str] = None):
        """
        Initialize the simple retriever with a knowledge base.
        
        Args:
            knowledge_base: Dictionary mapping questions/queries to answers
        """
        self.knowledge_base = knowledge_base or {}
    
    def add_entry(self, query: str, answer: str):
        """
        Add an entry to the knowledge base.
        
        Args:
            query: The query/question
            answer: The corresponding answer
        """
        self.knowledge_base[query.lower()] = answer
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve information from the simple knowledge base.
        
        Args:
            query: The query to search for
            
        Returns:
            The retrieved information as a string
        """
        # Simple exact match
        if query.lower() in self.knowledge_base:
            return self.knowledge_base[query.lower()]
        
        # Simple keyword matching
        for key, value in self.knowledge_base.items():
            if all(word.lower() in key for word in query.split()):
                return value
        
        return "I don't have information about that." 