from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import asyncio
import logging
import os
import json
from pathlib import Path
import time
from datetime import datetime
import google.generativeai as genai

# Import your RAG system
from src.core.rag_system import GeminiRAGAgent, initialize_agent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
rag_agents = {}  # Store RAG agents per session/API key
agent_cache_timeout = 3600  # 1 hour timeout for cached agents

def validate_google_api_key(api_key: str) -> tuple[bool, str]:
    """
    Validate Google API key by testing it with a simple request
    Returns (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"
    
    if not api_key.startswith('AIza') or len(api_key) < 30:
        return False, "Invalid API key format"
    
    try:
        # Test the API key with a simple embedding request
        genai.configure(api_key=api_key)
        
        # Try a minimal embedding to test API key
        result = genai.embed_content(
            model="models/embedding-001",
            content="test",
            task_type="RETRIEVAL_QUERY"
        )
        
        if result and 'embedding' in result:
            return True, "API key is valid"
        else:
            return False, "API key validation failed - no embedding returned"
            
    except Exception as e:
        error_msg = str(e).lower()
        if 'api key' in error_msg:
            return False, "Invalid API key - authentication failed"
        elif 'quota' in error_msg or 'limit' in error_msg:
            return False, "API quota exceeded - please check your Google Cloud billing"
        elif 'permission' in error_msg:
            return False, "API key doesn't have permission for Gemini API"
        else:
            return False, f"API key validation error: {str(e)}"

class RAGService:
    def __init__(self):
        self.agents = {}
        self.last_access = {}
    
    async def get_or_create_agent(self, api_key: str) -> GeminiRAGAgent:
        """Get existing agent or create new one with proper error handling"""
        # Validate API key first
        is_valid, error_msg = validate_google_api_key(api_key)
        if not is_valid:
            raise Exception(f"API Key Validation Failed: {error_msg}")
        
        key_hash = str(hash(api_key))  # Use hash for privacy
        
        # Check if agent exists and is still valid
        if key_hash in self.agents and key_hash in self.last_access:
            if time.time() - self.last_access[key_hash] < agent_cache_timeout:
                self.last_access[key_hash] = time.time()
                logger.info("Using cached RAG agent")
                return self.agents[key_hash]
        
        # Create new agent
        logger.info("Creating new RAG agent for session")
        try:
            # Pass the API key explicitly to initialize_agent
            agent = await initialize_agent(api_key)
            if agent:
                self.agents[key_hash] = agent
                self.last_access[key_hash] = time.time()
                logger.info("Successfully created and cached new RAG agent")
                return agent
            else:
                raise Exception("Failed to initialize RAG agent - agent is None")
        except Exception as e:
            logger.error(f"Error creating RAG agent: {e}")
            # Clean up any partial state
            if key_hash in self.agents:
                del self.agents[key_hash]
            if key_hash in self.last_access:
                del self.last_access[key_hash]
            raise e
    
    def cleanup_old_agents(self):
        """Remove old agents to free memory"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, last_time in self.last_access.items():
            if current_time - last_time > agent_cache_timeout:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.agents:
                del self.agents[key]
            if key in self.last_access:
                del self.last_access[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} old agents")

# Initialize service
rag_service = RAGService()

@app.route('/')
def index():
    """Serve the frontend HTML"""
    # Read the HTML file content (you'll need to save the frontend HTML as a file)
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>RAG Chat Backend</h1>
        <p>Backend is running! Please serve the frontend HTML file separately or place it in templates/index.html</p>
        <p>Available endpoints:</p>
        <ul>
            <li>POST /api/test_connection - Test API key</li>
            <li>POST /api/query - Query the RAG system</li>
            <li>GET /api/health - Health check</li>
        </ul>
        """

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_agents': len(rag_service.agents)
    })

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """Test if the API key is valid"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'success': False, 'error': 'API key is required'}), 400
        
        logger.info("Testing API key connection...")
        
        # Validate API key
        is_valid, error_msg = validate_google_api_key(api_key)
        
        if not is_valid:
            logger.warning(f"API key validation failed: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Try to create agent (this will test the full pipeline)
        async def test_agent():
            try:
                agent = await rag_service.get_or_create_agent(api_key)
                
                # Get some basic info about the knowledge base
                info = agent.embedding_system.get_collection_info()
                
                return {
                    'success': True, 
                    'message': 'API key is valid and RAG system is ready',
                    'knowledge_base_info': {
                        'document_count': info.get('document_count', 0),
                        'collection_name': info.get('collection_name', 'Unknown')
                    }
                }
            except Exception as e:
                logger.error(f"Error testing agent: {e}")
                return {'success': False, 'error': f'Agent creation failed: {str(e)}'}
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_agent())
        finally:
            loop.close()
        
        if result['success']:
            logger.info("API key test successful")
            return jsonify(result)
        else:
            logger.error(f"API key test failed: {result.get('error')}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in test_connection: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def query_rag():
    """Query the RAG system"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        api_key = data.get('api_key', '').strip()
        query = data.get('query', '').strip()
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Query parameters
        top_k = data.get('top_k', 5)
        similarity_threshold = data.get('similarity_threshold', 0.3)
        
        logger.info(f"Processing query: '{query[:50]}...'")
        
        async def process_query():
            try:
                # Get or create agent (this will validate API key)
                agent = await rag_service.get_or_create_agent(api_key)
                
                # Process query
                result = await agent.query(
                    query, 
                    top_k=top_k, 
                    similarity_threshold=similarity_threshold
                )
                
                # Format response
                return {
                    'success': True,
                    'response': result['response'],
                    'retrieved_docs_count': result['retrieved_docs_count'],
                    'processing_time': result['processing_time'],
                    'timestamp': result['timestamp'],
                    'sources': [
                        {
                            'filename': doc['metadata'].get('filename', 'Unknown'),
                            'similarity': doc['similarity'],
                            'is_chunked': doc['metadata'].get('is_chunked', False),
                            'chunk_info': f"Chunk {doc['metadata'].get('chunk_id', 1)}/{doc['metadata'].get('total_chunks', 1)}" if doc['metadata'].get('is_chunked') else None
                        }
                        for doc in result['retrieved_docs'][:3]  # Return top 3 sources
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return {'success': False, 'error': str(e)}
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_query())
        finally:
            loop.close()
        
        if result['success']:
            logger.info(f"Query processed successfully in {result.get('processing_time', 0):.2f}s")
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in query_rag: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for a specific agent"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        key_hash = str(hash(api_key))
        
        if key_hash in rag_service.agents:
            rag_service.agents[key_hash].clear_conversation_history()
            logger.info("Conversation history cleared")
            return jsonify({'success': True, 'message': 'History cleared'})
        else:
            return jsonify({'success': False, 'error': 'No active session found'}), 404
            
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_stats', methods=['POST'])
def get_stats():
    """Get statistics about the knowledge base"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        async def get_agent_stats():
            try:
                agent = await rag_service.get_or_create_agent(api_key)
                
                # Get collection info
                collection_info = agent.embedding_system.get_collection_info()
                
                # Get conversation history length
                history_length = len(agent.get_conversation_history())
                
                return {
                    'success': True,
                    'collection_name': collection_info.get('collection_name', 'Unknown'),
                    'document_count': collection_info.get('document_count', 0),
                    'embedding_dim': collection_info.get('embedding_dim', 0),
                    'model_name': collection_info.get('model_name', 'Unknown'),
                    'conversation_turns': history_length
                }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(get_agent_stats())
        finally:
            loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.before_request
def before_request():
    """Clean up old agents periodically"""
    if hasattr(app, '_last_cleanup'):
        if time.time() - app._last_cleanup > 300:  # Cleanup every 5 minutes
            rag_service.cleanup_old_agents()
            app._last_cleanup = time.time()
    else:
        app._last_cleanup = time.time()

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )