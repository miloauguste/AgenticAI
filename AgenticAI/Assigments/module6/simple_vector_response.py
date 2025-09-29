"""
Simplified Vector database integration - can work with or without Pinecone
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from config import settings
import pickle
import os

# Try to import required packages, provide fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using mock embeddings.")

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
    PINECONE_NEW_API = True
except (ImportError, Exception) as e:
    try:
        import pinecone  # Fallback to old package name
        HAS_PINECONE = True
        PINECONE_NEW_API = False
    except (ImportError, Exception):
        HAS_PINECONE = False
        PINECONE_NEW_API = False
        print(f"Warning: Pinecone not available ({str(e)[:50]}...). Using in-memory storage.")

class VectorStore:
    """Vector database manager with fallback to in-memory storage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = None
        self.index = None
        self.in_memory_store = {}  # Fallback storage
        self.persist_file = "vector_store_data.pkl"  # File to persist in-memory data
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_vector_db()
        self._load_persistent_data()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model with fallback"""
        try:
            if HAS_SENTENCE_TRANSFORMERS:
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.logger.info(f"Initialized embedding model: {settings.EMBEDDING_MODEL}")
            else:
                self.logger.warning("Using mock embedding model")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
    
    def _initialize_vector_db(self):
        """Initialize vector database with fallback"""
        try:
            if HAS_PINECONE and settings.PINECONE_API_KEY:
                if PINECONE_NEW_API:
                    # Use new Pinecone API (v7.0.0+)
                    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                    
                    # Create or connect to index
                    existing_indexes = [index.name for index in pc.list_indexes()]
                    if settings.PINECONE_INDEX_NAME not in existing_indexes:
                        from pinecone import ServerlessSpec
                        pc.create_index(
                            name=settings.PINECONE_INDEX_NAME,
                            dimension=settings.VECTOR_DIMENSION,
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )
                    
                    self.index = pc.Index(settings.PINECONE_INDEX_NAME)
                    self.logger.info("Connected to Pinecone (v7.x API)")
                else:
                    # Use legacy API
                    import pinecone
                    pinecone.init(
                        api_key=settings.PINECONE_API_KEY,
                        environment=settings.PINECONE_ENVIRONMENT
                    )
                    
                    # Create or connect to index
                    if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                        pinecone.create_index(
                            name=settings.PINECONE_INDEX_NAME,
                            dimension=settings.VECTOR_DIMENSION,
                            metric='cosine'
                        )
                    
                    self.index = pinecone.Index(settings.PINECONE_INDEX_NAME)
                    self.logger.info("Connected to Pinecone (legacy API)")
            else:
                self.logger.info("Using in-memory vector storage - Pinecone not configured")
        except Exception as e:
            self.logger.error(f"Vector DB initialization failed: {str(e)}")
            self.logger.info("Falling back to in-memory storage")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            if self.embedding_model:
                return self.embedding_model.encode(texts, convert_to_numpy=True)
            else:
                # Mock embeddings
                return np.random.random((len(texts), settings.VECTOR_DIMENSION))
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            # Return mock embeddings as fallback
            return np.random.random((len(texts), settings.VECTOR_DIMENSION))
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert or update chunks in the vector database"""
        try:
            if not chunks:
                return {'successful_upserts': 0, 'failed_upserts': 0}
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            upserted_count = 0
            
            if self.index:  # Pinecone
                vectors_to_upsert = []
                for i, chunk in enumerate(chunks):
                    vector_id = f"chunk_{chunk.get('chunk_id', i)}"
                    metadata = self._prepare_metadata(chunk)
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embeddings[i].tolist(),
                        'metadata': metadata
                    })
                
                # Upsert to Pinecone
                upsert_response = self.index.upsert(vectors_to_upsert)
                upserted_count = upsert_response.get('upserted_count', len(vectors_to_upsert))
                
            else:  # In-memory storage
                for i, chunk in enumerate(chunks):
                    vector_id = f"chunk_{chunk.get('chunk_id', i)}"
                    self.in_memory_store[vector_id] = {
                        'embedding': embeddings[i],
                        'text': chunk.get('text', ''),
                        'metadata': chunk.get('metadata', {})
                    }
                    upserted_count += 1
                
                # Save persistent data after adding new vectors
                self._save_persistent_data()
            
            return {
                'successful_upserts': upserted_count,
                'failed_upserts': len(chunks) - upserted_count,
                'upsert_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to upsert chunks: {str(e)}")
            return {'successful_upserts': 0, 'failed_upserts': len(chunks), 'error': str(e)}
    
    def search_similar(self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            all_results = []
            
            # Search Pinecone if available
            if self.index:  # Pinecone
                search_response = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k * 2,  # Get more results to ensure good ranking after merging
                    include_metadata=True,
                    filter=filter_metadata
                )
                
                for match in search_response.get('matches', []):
                    all_results.append({
                        'id': match['id'],
                        'score': match['score'],
                        'text': match.get('metadata', {}).get('text', ''),
                        'metadata': match.get('metadata', {}),
                        'source': 'pinecone'
                    })
            
            # Also search in-memory store (this contains PDFs and other vectors)
            for vector_id, stored_data in self.in_memory_store.items():
                # Calculate cosine similarity
                stored_embedding = stored_data['embedding']
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                all_results.append({
                    'id': vector_id,
                    'score': float(similarity),
                    'text': stored_data['text'],
                    'metadata': stored_data['metadata'],
                    'source': 'memory'
                })
            
            # Remove duplicates based on ID (prefer in-memory version if both exist)
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    # Remove the 'source' field before returning
                    result_copy = {k: v for k, v in result.items() if k != 'source'}
                    unique_results.append(result_copy)
            
            # Sort by score and return top_k
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            
            return unique_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to search: {str(e)}")
            return []
    
    def search_hybrid(self, query: str, keywords: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword matching"""
        # For simplified version, just do semantic search
        return self.search_similar(query, top_k)
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by metadata criteria"""
        results = []
        
        if self.index:  # Pinecone
            # Use dummy vector for metadata-only search
            dummy_vector = [0.0] * settings.VECTOR_DIMENSION
            search_response = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter
            )
            
            for match in search_response.get('matches', []):
                results.append({
                    'id': match['id'],
                    'text': match.get('metadata', {}).get('text', ''),
                    'metadata': match.get('metadata', {})
                })
        
        else:  # In-memory search
            for vector_id, stored_data in self.in_memory_store.items():
                metadata = stored_data['metadata']
                
                # Simple metadata matching
                match = True
                for key, value in metadata_filter.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    results.append({
                        'id': vector_id,
                        'text': stored_data['text'],
                        'metadata': metadata
                    })
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            if self.index:  # Pinecone
                stats = self.index.describe_index_stats()
                return {
                    'total_vector_count': stats.get('total_vector_count', 0),
                    'dimension': stats.get('dimension', settings.VECTOR_DIMENSION),
                    'index_fullness': stats.get('index_fullness', 0.0),
                    'index_name': settings.PINECONE_INDEX_NAME
                }
            else:  # In-memory
                return {
                    'total_vector_count': len(self.in_memory_store),
                    'dimension': settings.VECTOR_DIMENSION,
                    'index_fullness': 0.0,
                    'index_name': 'in_memory'
                }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {str(e)}")
            return {'total_vector_count': 0, 'dimension': settings.VECTOR_DIMENSION}
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear all vectors from the index"""
        try:
            if self.index:
                self.index.delete(delete_all=True)
            else:
                self.in_memory_store.clear()
                self._save_persistent_data()
            
            return {
                'action': 'cleared_index',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to clear index: {str(e)}")
            raise
    
    def _prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for storage"""
        metadata = {
            'chunk_id': chunk.get('chunk_id'),
            'chunk_type': chunk.get('chunk_type'),
            'word_count': chunk.get('word_count'),
            'char_count': chunk.get('char_count'),
            'text': chunk.get('text', '')[:1000],  # Store first 1000 chars
            'created_at': chunk.get('created_at', datetime.now().isoformat())
        }
        
        # Add chunk-specific metadata
        chunk_metadata = chunk.get('metadata', {})
        if chunk_metadata:
            metadata.update(chunk_metadata)
        
        return metadata
    
    def _load_persistent_data(self):
        """Load persistent in-memory data from disk"""
        try:
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'rb') as f:
                    self.in_memory_store = pickle.load(f)
                self.logger.info(f"Loaded {len(self.in_memory_store)} vectors from persistent storage")
            else:
                self.logger.info("No persistent data found, starting with empty vector store")
        except Exception as e:
            self.logger.error(f"Failed to load persistent data: {str(e)}")
            self.in_memory_store = {}
    
    def _save_persistent_data(self):
        """Save persistent in-memory data to disk"""
        try:
            with open(self.persist_file, 'wb') as f:
                pickle.dump(self.in_memory_store, f)
            self.logger.info(f"Saved {len(self.in_memory_store)} vectors to persistent storage")
        except Exception as e:
            self.logger.error(f"Failed to save persistent data: {str(e)}")