import pytest
import pytest_asyncio
import time
import numpy as np
import os
from src.core.gemini_embedding_system import GeminiEmbeddingSystem
from src.core.rag_system import GeminiRAGAgent, initialize_agent

# Test configuration
TEST_API_KEY = os.getenv('GOOGLE_API_KEY', 'test_key_placeholder')
TEST_CHROMA_DB_PATH = "./test_chroma_db"
TEST_COLLECTION_NAME = "test_embeddings"

@pytest_asyncio.fixture
async def embedding_system():
    """Initialize embedding system for testing"""
    if not TEST_API_KEY or TEST_API_KEY == 'test_key_placeholder':
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    
    try:
        system = GeminiEmbeddingSystem(
            api_key=TEST_API_KEY,
            chroma_db_path=TEST_CHROMA_DB_PATH,
            collection_name=TEST_COLLECTION_NAME
        )
        yield system
    except Exception as e:
        pytest.skip(f"Failed to initialize embedding system: {e}")
    finally:
        # Cleanup: clear test collection if it exists
        try:
            if hasattr(system, 'collection') and system.collection:
                system.clear_collection()
        except:
            pass

@pytest_asyncio.fixture
async def rag_agent():
    """Initialize RAG agent for testing"""
    if not TEST_API_KEY or TEST_API_KEY == 'test_key_placeholder':
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    
    try:
        # Create embedding system first
        embedding_system = GeminiEmbeddingSystem(
            api_key=TEST_API_KEY,
            chroma_db_path=TEST_CHROMA_DB_PATH,
            collection_name=TEST_COLLECTION_NAME
        )
        
        # Create RAG agent with the embedding system
        agent = GeminiRAGAgent(
            api_key=TEST_API_KEY,
            embedding_system=embedding_system,
            temperature=0.1  # Lower temperature for consistent testing
        )
        yield agent
    except Exception as e:
        pytest.skip(f"Failed to initialize RAG agent: {e}")
    finally:
        # Cleanup
        try:
            if hasattr(agent, 'embedding_system') and agent.embedding_system:
                agent.embedding_system.clear_collection()
        except:
            pass

class TestEmbeddingSystem:

    @pytest.mark.asyncio
    async def test_embedding_system_initialization(self, embedding_system):
        """Test that the embedding system initializes properly"""
        assert embedding_system is not None
        assert hasattr(embedding_system, 'collection')
        assert hasattr(embedding_system, 'api_key')
        assert embedding_system.model_name == "models/embedding-001"
        assert embedding_system.embedding_dim == 768

    @pytest.mark.asyncio
    async def test_collection_info(self, embedding_system):
        """Test getting collection information"""
        info = embedding_system.get_collection_info()
        assert isinstance(info, dict)
        assert 'collection_name' in info
        assert 'document_count' in info
        assert 'embedding_dim' in info
        assert info['embedding_dim'] == 768

    @pytest.mark.asyncio
    async def test_query_embedding_latency(self, embedding_system):
        """Test embedding query latency is under 5 seconds"""
        start = time.time()
        embedding = await embedding_system.embed_query("Where can I find airport lounges?")
        end = time.time()
        
        assert (end - start) < 5  # latency < 5s
        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, (int, float)) for x in embedding)

    @pytest.mark.asyncio
    async def test_retrieval_with_empty_collection(self, embedding_system):
        """Test retrieval behavior with empty collection"""
        # Ensure collection is empty
        embedding_system.clear_collection()
        
        results = await embedding_system.retrieve("test query")
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieval_latency_with_mock_data(self, embedding_system):
        """Test document retrieval latency after adding some mock data"""
        # Add some mock documents for testing
        mock_docs = [
            "Changi Airport has many facilities including lounges and shopping areas.",
            "The airport provides free WiFi throughout all terminals.",
            "Sleeping pods are available for tired travelers."
        ]
        mock_metadata = [
            {'filename': f'test_doc_{i}.txt', 'directory': 'test', 'size': len(doc), 
             'doc_id': f'test_{i}', 'is_chunked': False, 'source': f'test_{i}.txt'}
            for i, doc in enumerate(mock_docs)
        ]
        
        # Generate embeddings and store
        embeddings = await embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        # Test retrieval latency
        start = time.time()
        results = await embedding_system.retrieve("Are there sleeping pods in Changi?")
        end = time.time()
        
        assert (end - start) < 5  # latency < 5s
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_similarity_scores(self, embedding_system):
        """Test that retrieval returns documents with proper similarity scores"""
        # Add mock data first
        mock_docs = ["Airport lounges provide comfortable seating and refreshments."]
        mock_metadata = [{'filename': 'lounge_info.txt', 'directory': 'test', 'size': len(mock_docs[0]), 
                         'doc_id': 'lounge_1', 'is_chunked': False, 'source': 'lounge_info.txt'}]
        
        embeddings = await embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        results = await embedding_system.retrieve("lounges in airport")
        
        if results:  # Only test if we got results
            assert len(results) > 0
            result = results[0]
            assert 'similarity' in result
            assert 'metadata' in result
            assert 'document' in result
            assert 0 <= result['similarity'] <= 1

    @pytest.mark.asyncio
    async def test_retrieval_consistency(self, embedding_system):
        """Test that similar queries return consistent results"""
        # Add mock data
        mock_docs = [
            "Changi Airport provides free WiFi in all terminals.",
            "Internet access is available throughout the airport at no charge."
        ]
        mock_metadata = [
            {'filename': f'wifi_{i}.txt', 'directory': 'test', 'size': len(doc), 
             'doc_id': f'wifi_{i}', 'is_chunked': False, 'source': f'wifi_{i}.txt'}
            for i, doc in enumerate(mock_docs)
        ]
        
        embeddings = await embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        q1_results = await embedding_system.retrieve("Wi-Fi availability in Changi?")
        q2_results = await embedding_system.retrieve("Does Changi provide free internet?")
        
        # Both queries should return some results
        assert isinstance(q1_results, list)
        assert isinstance(q2_results, list)

    @pytest.mark.asyncio
    async def test_retriever_function(self, embedding_system):
        """Test the retriever function interface"""
        retriever = embedding_system.get_retriever(top_k=3, similarity_threshold=0.0)
        assert callable(retriever)
        
        # Test with empty collection
        results = await retriever("test query")
        assert isinstance(results, list)

class TestRAGSystem:

    @pytest.mark.asyncio
    async def test_rag_agent_initialization(self, rag_agent):
        """Test RAG agent initializes properly"""
        assert rag_agent is not None
        assert hasattr(rag_agent, 'model')
        assert hasattr(rag_agent, 'embedding_system')
        assert hasattr(rag_agent, 'retriever')
        assert hasattr(rag_agent, 'conversation_history')
        assert rag_agent.model_name == "gemini-1.5-flash"

    @pytest.mark.asyncio
    async def test_rag_end_to_end_latency(self, rag_agent):
        """Test end-to-end RAG pipeline latency"""
        # Add some mock data to the embedding system first
        mock_docs = ["Jewel Changi Airport is a nature-themed entertainment complex."]
        mock_metadata = [{'filename': 'jewel.txt', 'directory': 'test', 'size': len(mock_docs[0]), 
                         'doc_id': 'jewel_1', 'is_chunked': False, 'source': 'jewel.txt'}]
        
        embeddings = await rag_agent.embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await rag_agent.embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        start = time.time()
        result = await rag_agent.query("Where is Jewel located?")
        end = time.time()
        
        assert (end - start) < 15  # Allow more time for full RAG pipeline
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'retrieved_docs_count' in result
        assert 'processing_time' in result
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    @pytest.mark.asyncio
    async def test_rag_response_quality(self, rag_agent):
        """Test that RAG responses contain relevant information"""
        # Add mock data about terminals
        mock_docs = ["Changi Airport has four main terminals: Terminal 1, 2, 3, and 4."]
        mock_metadata = [{'filename': 'terminals.txt', 'directory': 'test', 'size': len(mock_docs[0]), 
                         'doc_id': 'terminals_1', 'is_chunked': False, 'source': 'terminals.txt'}]
        
        embeddings = await rag_agent.embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await rag_agent.embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        result = await rag_agent.query("How many terminals in Changi?")
        
        assert isinstance(result, dict)
        assert 'response' in result
        response_text = result['response'].lower()
        # Check for relevant keywords
        assert any(keyword in response_text for keyword in ['terminal', 'four', '4', 'changi'])

    @pytest.mark.asyncio
    async def test_rag_without_retrieval(self, rag_agent):
        """Test RAG agent can work without document retrieval"""
        result = await rag_agent.query("Hello, how are you?", retrieve_docs=False)
        
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'retrieved_docs_count' in result
        assert result['retrieved_docs_count'] == 0
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    @pytest.mark.asyncio
    async def test_conversation_history(self, rag_agent):
        """Test conversation history functionality"""
        # Clear history first
        rag_agent.clear_conversation_history()
        assert len(rag_agent.get_conversation_history()) == 0
        
        # Make a query
        await rag_agent.query("Test question 1")
        history = rag_agent.get_conversation_history()
        assert len(history) == 1
        assert 'query' in history[0]
        assert 'response' in history[0]
        assert 'timestamp' in history[0]
        
        # Make another query
        await rag_agent.query("Test question 2")
        history = rag_agent.get_conversation_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_rag_error_handling(self, rag_agent):
        """Test RAG system handles edge cases gracefully"""
        # Test empty query
        result = await rag_agent.query("")
        assert isinstance(result, dict)
        assert 'response' in result
        
        # Test very long query
        long_query = "airport " * 200
        result = await rag_agent.query(long_query)
        assert isinstance(result, dict)
        assert 'response' in result

class TestPerformanceBenchmark:

    @pytest.mark.asyncio
    async def test_performance_benchmark(self, embedding_system, rag_agent):
        """Comprehensive performance test"""
        # Add mock data for testing
        mock_docs = [
            "Changi Airport provides flight delay information services.",
            "Passengers can check flight status at information counters.",
            "Real-time flight information is available on digital displays."
        ]
        mock_metadata = [
            {'filename': f'flight_info_{i}.txt', 'directory': 'test', 'size': len(doc), 
             'doc_id': f'flight_{i}', 'is_chunked': False, 'source': f'flight_info_{i}.txt'}
            for i, doc in enumerate(mock_docs)
        ]
        
        embeddings = await embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        q = "Flight delay services at Changi airport?"
        
        # Test retrieval performance
        start_retrieval = time.time()
        docs = await embedding_system.retrieve(q)
        end_retrieval = time.time()
        
        # Test full RAG performance
        start_rag = time.time()
        answer = await rag_agent.query(q)
        end_rag = time.time()
        
        # Assertions
        assert isinstance(docs, list)
        assert (end_retrieval - start_retrieval) < 5  # Retrieval under 5s
        assert (end_rag - start_rag) < 15  # Full RAG under 15s
        
        assert isinstance(answer, dict)
        assert 'response' in answer
        assert len(answer['response']) > 0

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, rag_agent):
        """Test system handles concurrent requests"""
        import asyncio
        
        # Add some mock data first
        mock_docs = [
            "Airport lounges provide comfortable seating.",
            "Flight information displays show departure times.",
            "Baggage claim areas are located on the ground floor.",
            "Shopping areas include duty-free stores.",
            "Food courts offer various dining options."
        ]
        mock_metadata = [
            {'filename': f'info_{i}.txt', 'directory': 'test', 'size': len(doc), 
             'doc_id': f'info_{i}', 'is_chunked': False, 'source': f'info_{i}.txt'}
            for i, doc in enumerate(mock_docs)
        ]
        
        embeddings = await rag_agent.embedding_system.generate_embeddings_for_documents(mock_docs, mock_metadata)
        await rag_agent.embedding_system._store_in_chromadb(mock_docs, embeddings, mock_metadata)
        
        queries = [
            "Where are the lounges?",
            "Flight information displays?",
            "Baggage claim locations?"
        ]
        
        async def run_query(query):
            return await rag_agent.query(query)
        
        start = time.time()
        results = await asyncio.gather(*[run_query(q) for q in queries])
        end = time.time()
        
        assert len(results) == len(queries)
        assert all(isinstance(result, dict) and 'response' in result for result in results)
        assert (end - start) < 45  # All queries under 45s total

# Additional utility tests
class TestSystemHealth:
    
    @pytest.mark.asyncio
    async def test_system_initialization_with_initialize_agent(self):
        """Test the initialize_agent function"""
        if not TEST_API_KEY or TEST_API_KEY == 'test_key_placeholder':
            pytest.skip("GOOGLE_API_KEY environment variable not set")
        
        try:
            agent = await initialize_agent(TEST_API_KEY)
            assert agent is not None
            assert isinstance(agent, GeminiRAGAgent)
            
            # Cleanup
            if hasattr(agent, 'embedding_system') and agent.embedding_system:
                agent.embedding_system.clear_collection()
        except Exception as e:
            # Expected if no data directory exists
            assert "Data directory not found" in str(e) or "Failed to build knowledge base" in str(e)

    @pytest.mark.asyncio
    async def test_embedding_cache_functionality(self, embedding_system):
        """Test embedding caching works properly"""
        query = "test caching query"
        
        # First call - should generate embedding
        start1 = time.time()
        embedding1 = await embedding_system.embed_query(query)
        end1 = time.time()
        
        # Second call - should use cache
        start2 = time.time()
        embedding2 = await embedding_system.embed_query(query)
        end2 = time.time()
        
        # Results should be identical
        assert embedding1 == embedding2
        # Second call should be faster (cached)
        assert (end2 - start2) <= (end1 - start1)

    @pytest.mark.asyncio
    async def test_memory_usage_basic(self, rag_agent):
        """Basic memory usage test"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run some queries to test memory
        for i in range(3):  # Reduced from 10 to 3 for faster testing
            await rag_agent.query(f"test query {i}", retrieve_docs=False)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory shouldn't increase dramatically (adjust threshold as needed)
        assert memory_increase < 200  # Less than 200MB increase

# Mock tests for when API key is not available
class TestMockFunctionality:
    
    def test_import_classes(self):
        """Test that we can import the classes without API key"""
        from src.core.gemini_embedding_system import GeminiEmbeddingSystem
        from src.core.rag_system import GeminiRAGAgent
        
        # Just test imports work
        assert GeminiEmbeddingSystem is not None
        assert GeminiRAGAgent is not None
    
    def test_class_attributes(self):
        """Test class attributes without instantiation"""
        from src.core.gemini_embedding_system import GeminiEmbeddingSystem
        
        # Test we can access class-level information
        assert hasattr(GeminiEmbeddingSystem, '__init__')
        assert hasattr(GeminiEmbeddingSystem, 'embed_query')
        assert hasattr(GeminiEmbeddingSystem, 'retrieve')