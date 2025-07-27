import os
import asyncio
import numpy as np
from typing import List, Union, Tuple, Dict, Any, Optional
import google.generativeai as genai
import logging
import pickle
from pathlib import Path
import json
import glob
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm
import time
import chromadb
from chromadb.config import Settings
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbeddingSystem:
    def __init__(self, 
                 api_key: str = None,
                 cache_dir: str = "./embedding_cache",
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "gemini_embeddings"):
        
        # Handle API key configuration
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")   
        
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Store API key for re-use
        self.api_key = api_key
        
        # Configure Gemini API
        try:
            genai.configure(api_key=api_key)
            logger.info("Successfully configured Gemini API for embedding system")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise ValueError(f"Failed to configure Gemini API: {e}")
        
        # Test API key with a simple request
        try:
            logger.info("Testing API key with embedding model...")
            test_result = genai.embed_content(
                model="models/embedding-001",
                content="test",
                task_type="RETRIEVAL_QUERY"
            )
            if not test_result or 'embedding' not in test_result:
                raise Exception("API key test failed - no embedding returned")
            logger.info("API key test successful")
        except Exception as e:
            logger.error(f"API key test failed: {e}")
            raise ValueError(f"Invalid API key or API access issue: {e}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model_name = "models/embedding-001"
        self._embedding_dim = 768
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Successfully connected to ChromaDB at: {chroma_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise ValueError(f"Failed to initialize ChromaDB: {e}")
        
        self.collection_name = collection_name
        self.collection = None
        
        self.embedding_cache = {}
        self._load_cache()
        
        self._initialize_chroma_collection()
    
    def _initialize_chroma_collection(self):
        """Initialize or get ChromaDB collection"""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll provide embeddings manually
            )
            logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            logger.info(f"Collection contains {self.collection.count()} documents")
        except Exception as get_error:
            # Collection doesn't exist, create it
            try:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create ChromaDB collection: {create_error}")
                raise ValueError(f"Failed to initialize ChromaDB collection: {create_error}")
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def _load_cache(self):
        cache_file = self.cache_dir / "gemini_embedding_cache.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}
        else:
            logger.info("Starting with empty embedding cache")
            self.embedding_cache = {}
    
    def _save_cache(self):
        cache_file = self.cache_dir / "gemini_embedding_cache.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        return f"{self.model_name}:{hash(text.strip())}"
    
    def _chunk_text(self, text: str, max_bytes: int = 30000, overlap: int = 200) -> List[str]:
        """Split text into chunks that fit within API limits"""
        # Convert to bytes to check actual size
        text_bytes = text.encode('utf-8')
        
        if len(text_bytes) <= max_bytes:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_bytes = len((word + ' ').encode('utf-8'))
            
            if current_size + word_bytes > max_bytes and current_chunk:
                # Create chunk with overlap
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + [word]
                current_size = len(' '.join(current_chunk).encode('utf-8'))
            else:
                current_chunk.append(word)
                current_size += word_bytes
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def _embed_with_gemini(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} texts with task_type: {task_type}")
        embeddings = []
        failed_count = 0
        oversized_count = 0
        
        # Ensure API is configured with our key
        genai.configure(api_key=self.api_key)
        
        for i, text in enumerate(texts):
            try:
                # Check text size before embedding
                text_bytes = len(text.encode('utf-8'))
                if text_bytes > 35000:  # Leave some buffer
                    logger.warning(f"Text {i+1} still too large ({text_bytes} bytes), truncating")
                    # Truncate text to fit
                    text = text[:25000] + "..."
                    oversized_count += 1
                
                if i > 0:
                    await asyncio.sleep(0.1)  # Rate limiting
                
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.model_name,
                    content=text,
                    task_type=task_type
                )
                
                if result and 'embedding' in result:
                    embeddings.append(result['embedding'])
                else:
                    logger.error(f"No embedding returned for text {i+1}")
                    embeddings.append([0.0] * self._embedding_dim)
                    failed_count += 1
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to embed text {i+1}: {e}")
                embeddings.append([0.0] * self._embedding_dim)
                
                # Add longer delay after failures
                await asyncio.sleep(1.0)
        
        if oversized_count > 0:
            logger.warning(f"Truncated {oversized_count} oversized texts")
        
        if failed_count > 0:
            logger.warning(f"Failed to embed {failed_count}/{len(texts)} texts")
        
        logger.info(f"Completed embedding: {len(texts) - failed_count}/{len(texts)} successful")
        return embeddings
    
    def load_documents_from_directories(self, directories: List[str]) -> Tuple[List[str], List[Dict]]:
        documents = []
        metadata = []
        
        if not directories:
            logger.warning("No directories provided")
            return documents, metadata
        
        dir_pbar = sync_tqdm(directories, desc="Loading directories", unit="dir")
        
        for directory in dir_pbar:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
                
            dir_pbar.set_postfix_str(f"Processing {dir_path.name}")
            logger.info(f"Loading documents from: {directory}")
            
            txt_files = list(dir_path.glob("*.txt"))
            txt_files.sort()
            
            if not txt_files:
                logger.warning(f"No .txt files found in {directory}")
                continue
            
            file_pbar = sync_tqdm(txt_files, desc=f"Loading files from {dir_path.name}", 
                                leave=False, unit="file")
            
            total_chunks = 0
            large_files_chunked = 0
            
            for txt_file in file_pbar:
                file_pbar.set_postfix_str(txt_file.name)
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Check if content needs chunking
                            content_bytes = len(content.encode('utf-8'))
                            
                            if content_bytes > 30000:
                                # Chunk large documents
                                chunks = self._chunk_text(content)
                                large_files_chunked += 1
                                
                                for i, chunk in enumerate(chunks):
                                    documents.append(chunk)
                                    metadata.append({
                                        'source': str(txt_file),
                                        'filename': f"{txt_file.name}",
                                        'chunk_id': i + 1,
                                        'total_chunks': len(chunks),
                                        'directory': str(dir_path),
                                        'size': len(chunk),
                                        'original_size': len(content),
                                        'doc_id': str(uuid.uuid4()),
                                        'is_chunked': True
                                    })
                                total_chunks += len(chunks)
                                
                                logger.info(f"Chunked {txt_file.name}: {content_bytes} bytes → {len(chunks)} chunks")
                            else:
                                # Single document
                                documents.append(content)
                                metadata.append({
                                    'source': str(txt_file),
                                    'filename': txt_file.name,
                                    'directory': str(dir_path),
                                    'size': len(content),
                                    'doc_id': str(uuid.uuid4()),
                                    'is_chunked': False
                                })
                        else:
                            logger.warning(f"Skipping empty file: {txt_file.name}")
                except Exception as e:
                    logger.error(f"Failed to read file {txt_file}: {e}")
            
            logger.info(f"Loaded {len(txt_files)} files from {directory}")
            if large_files_chunked > 0:
                logger.info(f"Chunked {large_files_chunked} large files into {total_chunks} total chunks")
        
        logger.info(f"Total documents/chunks loaded: {len(documents)}")
        return documents, metadata
    
    async def generate_embeddings_for_documents(self, documents: List[str], metadata: List[Dict]) -> List[List[float]]:
        """Generate embeddings for documents, using cache when available"""
        if not documents:
            logger.warning("No documents provided for embedding")
            return []
        
        embeddings = []
        docs_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        logger.info("Checking embedding cache...")
        for i, doc in enumerate(documents):
            cache_key = self._get_cache_key(doc)
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                embeddings.append(None)
                docs_to_embed.append(doc)
                indices_to_embed.append(i)
        
        if docs_to_embed:
            logger.info(f"Generating embeddings for {len(docs_to_embed)} documents (cached: {len(documents) - len(docs_to_embed)})")
            try:
                new_embeddings = await self._embed_with_gemini(docs_to_embed, task_type="RETRIEVAL_DOCUMENT")
                
                # Fill in embeddings and cache them
                for i, (idx, new_embedding) in enumerate(zip(indices_to_embed, new_embeddings)):
                    embeddings[idx] = new_embedding
                    cache_key = self._get_cache_key(docs_to_embed[i])
                    self.embedding_cache[cache_key] = new_embedding
                
                self._save_cache()
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Fill remaining with zero vectors
                for idx in indices_to_embed:
                    if embeddings[idx] is None:
                        embeddings[idx] = [0.0] * self._embedding_dim
        else:
            logger.info("All embeddings found in cache")
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query for retrieval"""
        if not query.strip():
            logger.warning("Empty query provided")
            return [0.0] * self._embedding_dim
        
        logger.info(f"Embedding query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        cache_key = self._get_cache_key(f"query:{query}")
        if cache_key in self.embedding_cache:
            logger.info("Query embedding found in cache")
            return self.embedding_cache[cache_key]
        
        try:
            # Ensure API is configured
            genai.configure(api_key=self.api_key)
            
            embeddings = await self._embed_with_gemini([query], task_type="RETRIEVAL_QUERY")
            query_embedding = embeddings[0]
            
            # Cache the result
            self.embedding_cache[cache_key] = query_embedding
            return query_embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * self._embedding_dim
    
    async def build_vector_database(self, directories: List[str]) -> Dict[str, Any]:
        """Build vector database and store in ChromaDB"""
        logger.info("Building vector database with Gemini embeddings")
        
        try:
            # Load documents
            documents, metadata = self.load_documents_from_directories(directories)
            
            if not documents:
                logger.warning("No documents found in the specified directories")
                return {"total_documents": 0, "is_built": False, "error": "No documents found"}
            
            # Generate embeddings
            embeddings = await self.generate_embeddings_for_documents(documents, metadata)
            
            if not embeddings or all(emb == [0.0] * self._embedding_dim for emb in embeddings):
                logger.error("Failed to generate valid embeddings")
                return {"total_documents": 0, "is_built": False, "error": "Failed to generate embeddings"}
            
            # Store in ChromaDB
            await self._store_in_chromadb(documents, embeddings, metadata)
            
            result = {
                'is_built': True,
                'model_name': self.model_name,
                'embedding_dim': self._embedding_dim,
                'total_documents': len(documents),
                'collection_name': self.collection_name
            }
            
            logger.info(f"Vector database built successfully with {len(documents)} documents in ChromaDB")
            return result
            
        except Exception as e:
            logger.error(f"Error building vector database: {e}")
            return {"total_documents": 0, "is_built": False, "error": str(e)}
    
    async def _store_in_chromadb(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """Store documents and embeddings in ChromaDB"""
        logger.info(f"Storing {len(documents)} documents in ChromaDB")
        
        if not documents or not embeddings or not metadata:
            raise ValueError("Documents, embeddings, and metadata must all be provided")
        
        if len(documents) != len(embeddings) or len(documents) != len(metadata):
            raise ValueError("Documents, embeddings, and metadata must have the same length")
        
        # Prepare data for ChromaDB
        ids = [meta['doc_id'] for meta in metadata]
        
        # Convert metadata to strings for ChromaDB compatibility
        chroma_metadata = []
        for meta in metadata:
            chroma_meta = {
                'filename': meta['filename'],
                'directory': meta['directory'],
                'size': str(meta['size']),
                'source': meta['source'],
                'is_chunked': str(meta.get('is_chunked', False))
            }
            if meta.get('is_chunked', False):
                chroma_meta['chunk_id'] = str(meta.get('chunk_id', 1))
                chroma_meta['total_chunks'] = str(meta.get('total_chunks', 1))
                chroma_meta['original_size'] = str(meta.get('original_size', meta['size']))
            
            chroma_metadata.append(chroma_meta)
        
        # Clear existing collection if rebuilding
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(f"Clearing existing collection with {existing_count} documents")
                # Delete all documents
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                logger.info("Cleared existing ChromaDB collection")
        except Exception as e:
            logger.warning(f"Error clearing collection: {e}")
        
        # Add documents in batches to avoid memory issues
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        pbar = sync_tqdm(range(0, len(documents), batch_size), 
                        desc="Storing in ChromaDB", unit="batch")
        
        for i in pbar:
            end_idx = min(i + batch_size, len(documents))
            batch_documents = documents[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadata = chroma_metadata[i:end_idx]
            
            try:
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                pbar.set_postfix_str(f"Stored {end_idx}/{len(documents)} docs")
            except Exception as e:
                logger.error(f"Error storing batch {i//batch_size + 1}: {e}")
                raise e
        
        logger.info(f"Successfully stored {len(documents)} documents in ChromaDB")
    
    async def retrieve(self, 
                      query: str, 
                      top_k: int = 10,
                      similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB"""
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided for retrieval")
            return []
        
        logger.info(f"Retrieving for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        try:
            # Get query embedding
            query_embedding = await self.embed_query(query)
            
            if query_embedding == [0.0] * self._embedding_dim:
                logger.error("Failed to generate query embedding")
                return []
            
            # Query ChromaDB
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("No documents in collection")
                return []
            
            n_results = min(top_k, collection_count)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity (ChromaDB returns distances)
                    similarity = 1 - distance
                    
                    if similarity >= similarity_threshold:
                        # Convert string metadata back to appropriate types
                        processed_metadata = metadata.copy()
                        if metadata.get('is_chunked') == 'True':
                            processed_metadata['is_chunked'] = True
                            processed_metadata['chunk_id'] = int(metadata.get('chunk_id', 1))
                            processed_metadata['total_chunks'] = int(metadata.get('total_chunks', 1))
                            processed_metadata['original_size'] = int(metadata.get('original_size', metadata['size']))
                        else:
                            processed_metadata['is_chunked'] = False
                        
                        processed_metadata['size'] = int(metadata['size'])
                        
                        retrieved_docs.append({
                            'document': doc,
                            'metadata': processed_metadata,
                            'similarity': float(similarity),
                            'distance': float(distance)
                        })
            
            # Filter by similarity threshold
            filtered_results = [doc for doc in retrieved_docs if doc['similarity'] >= similarity_threshold]
            
            logger.info(f"Retrieved {len(filtered_results)} documents above threshold {similarity_threshold}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []
    
    def get_retriever(self, top_k: int = 10, similarity_threshold: float = 0.0):
        """Get a retriever function for RAG applications"""
        logger.info(f"Creating retriever with top_k: {top_k}, threshold: {similarity_threshold}")
        
        async def retriever(query: str, k: Optional[int] = None, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
            actual_k = k if k is not None else top_k
            actual_threshold = threshold if threshold is not None else similarity_threshold
            
            return await self.retrieve(query, top_k=actual_k, similarity_threshold=actual_threshold)
        
        return retriever
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dim": self._embedding_dim,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self):
        """Clear all documents from the ChromaDB collection"""
        if self.collection:
            try:
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                logger.info("Cleared ChromaDB collection")
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")
    
    def __del__(self):
        if hasattr(self, 'embedding_cache') and self.embedding_cache:
            try:
                self._save_cache()
            except Exception as e:
                logger.error(f"Error saving cache during cleanup: {e}")

async def test_gemini_embedding_system():
    """Test the Gemini embedding system"""
    try:
        # Get API key from environment or user input
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            api_key = input("Enter your Google API key: ").strip()
        
        if not api_key:
            logger.error("API key is required for testing")
            return
        
        embedding_system = GeminiEmbeddingSystem(
            api_key=api_key,
            chroma_db_path="./chroma_db",
            collection_name="website_embeddings"
        )
    except ValueError as e:
        logger.error(f"Error initializing embedding system: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return
    
    directories = ["./cleaned_data"]
    
    # Check if directories exist
    for directory in directories:
        if not Path(directory).exists():
            logger.error(f"Test directory not found: {directory}")
            logger.info("Please create the directory and add some .txt files for testing")
            return
    
    logger.info("Building vector database with ChromaDB storage")
    try:
        result = await embedding_system.build_vector_database(directories)
        
        if not result['is_built']:
            logger.error(f"Failed to build vector database: {result.get('error', 'Unknown error')}")
            return
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        return
    
    # Get collection info
    info = embedding_system.get_collection_info()
    logger.info(f"ChromaDB Collection Info: {info}")
    
    try:
        retriever = embedding_system.get_retriever(top_k=5, similarity_threshold=0.3)
        
        test_queries = [
            "What are the facilities at Changi Airport?",
            "Tell me about Jewel Changi Airport",
            "What shopping options are available?",
            "How can I get to the airport?",
            "What entertainment is available?"
        ]
        
        logger.info("Testing retriever functionality with ChromaDB")
        
        for query in test_queries:
            logger.info(f"Query: {query}")
            
            try:
                results = await retriever(query)
                
                logger.info(f"Retrieved {len(results)} documents:")
                for i, result in enumerate(results):
                    filename = result['metadata'].get('filename', 'Unknown')
                    similarity = result['similarity']
                    logger.info(f"{i+1}. Similarity: {similarity:.4f} - Source: {filename}")
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        logger.info("Testing different retriever parameters")
        
        query = "Changi Airport facilities"
        logger.info(f"Query: {query}")
        
        for k in [3, 5, 10]:
            try:
                results = await retriever(query, k=k)
                logger.info(f"Top {k} results: Retrieved {len(results)} documents")
            except Exception as e:
                logger.error(f"Error testing top_{k} retrieval: {e}")
        
        # Final summary
        final_info = embedding_system.get_collection_info()
        logger.info("Test Summary:")
        logger.info(f"ChromaDB Collection: {final_info.get('collection_name', 'Unknown')}")
        logger.info(f"Total documents: {final_info.get('document_count', 0)}")
        logger.info(f"Embedding dimension: {final_info.get('embedding_dim', 'Unknown')}")
        logger.info(f"Model: {final_info.get('model_name', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_embedding_system())