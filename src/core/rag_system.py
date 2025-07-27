import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from pathlib import Path
import json
import time
from datetime import datetime

# Import the embedding system from the previous file
from src.core.gemini_embedding_system import GeminiEmbeddingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiRAGAgent:
    """
    RAG Agent using Gemini 1.5 Flash for generation and ChromaDB for retrieval
    """
    
    def __init__(self, 
                 api_key: str = None,
                 embedding_system: GeminiEmbeddingSystem = None,
                 model_name: str = "gemini-1.5-flash",
                 max_context_length: int = 8000,
                 temperature: float = 0.7):
        
        if not api_key:
            raise ValueError("Google API key is required.")
        
        # Store API key for potential re-use
        self.api_key = api_key
        
        # Configure Gemini with the provided API key
        try:
            genai.configure(api_key=api_key)
            logger.info("Successfully configured Gemini API")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise ValueError(f"Failed to configure Gemini API with provided key: {e}")
        
        # Initialize Gemini model
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
            logger.info(f"Successfully initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise ValueError(f"Failed to initialize Gemini model {model_name}: {e}")
        
        # Initialize or use existing embedding system
        if embedding_system is None:
            try:
                logger.info("Creating new embedding system...")
                self.embedding_system = GeminiEmbeddingSystem(
                    api_key=api_key,  # Explicitly pass the API key
                    chroma_db_path="./chroma_db",
                    collection_name="website_embeddings"
                )
                logger.info("Successfully created embedding system")
            except Exception as e:
                logger.error(f"Failed to create embedding system: {e}")
                raise ValueError(f"Failed to create embedding system: {e}")
        else:
            self.embedding_system = embedding_system
            logger.info("Using provided embedding system")
        
        # Get retriever
        try:
            self.retriever = self.embedding_system.get_retriever(
                top_k=5, 
                similarity_threshold=0.3
            )
            logger.info("Successfully created retriever")
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise ValueError(f"Failed to create retriever: {e}")
        
        self.max_context_length = max_context_length
        
        # Conversation history
        self.conversation_history = []
        
        logger.info(f"Successfully initialized Gemini RAG Agent with model: {model_name}")
    
    def _create_system_prompt(self) -> str:
        """Create the enhanced system prompt for the RAG agent"""
        return """You are a helpful AI assistant with access to a comprehensive knowledge base about airports, facilities, and travel information. 

Your role is to:
1. Answer questions accurately using the provided context from the knowledge base
2. Structure your responses in a clear, organized manner using bullet points when appropriate
3. Provide specific details when available (locations, times, contact info, etc.)
4. Be conversational yet informative
5. If the context doesn't contain relevant information, clearly state this

Response Structure Guidelines:
- Use bullet points (•) for listing multiple items or features
- Use numbered lists (1., 2., 3.) for sequential steps or processes
- Group related information together
- Include specific details like terminal numbers, levels, and locations when available
- Cite sources when referencing specific information

Formatting Examples:
• For facilities: List each facility with its location and key features
• For food options: Organize by cuisine type or terminal location
• For procedures: Provide step-by-step instructions
• For general information: Use clear paragraphs with bullet points for key details

Guidelines:
- Always base your response on the provided context
- If you're unsure about something, acknowledge the uncertainty
- Focus on being helpful, accurate, and well-structured
- Use a friendly, professional tone
- When listing items, be comprehensive but concise"""
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context for the LLM"""
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc['metadata']
            source = metadata.get('filename', 'Unknown source')
            similarity = doc['similarity']
            content = doc['document']
            
            # Handle chunked documents
            if metadata.get('is_chunked', False):
                chunk_info = f" (Chunk {metadata.get('chunk_id', '?')}/{metadata.get('total_chunks', '?')})"
                source += chunk_info
            
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {similarity:.3f})]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _truncate_context(self, context: str, max_length: int) -> str:
        """Truncate context if it's too long"""
        if len(context) <= max_length:
            return context
        
        # Truncate and add indicator
        truncated = context[:max_length - 50]
        return truncated + "\n\n[... context truncated due to length ...]"
    
    def _create_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Create the full prompt for the LLM"""
        system_prompt = self._create_system_prompt()
        
        # Add conversation history if available
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                history_parts.append(f"User: {turn['query']}")
                history_parts.append(f"Assistant: {turn['response']}")
            
            if history_parts:
                history_text = f"\n\nRecent conversation:\n" + "\n".join(history_parts) + "\n"
        
        # Truncate context if needed
        available_length = self.max_context_length - len(system_prompt) - len(history_text) - len(query) - 200
        context = self._truncate_context(context, available_length)
        
        prompt = f"""{system_prompt}

Context from knowledge base:
{context}{history_text}

User Question: {query}

Please provide a helpful, well-structured response based on the context above. Use bullet points and clear organization when appropriate."""
        
        return prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using Gemini model"""
        try:
            logger.info(f"Generating response with {self.model_name}")
            
            # Ensure API is configured with our key
            genai.configure(api_key=self.api_key)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=self.generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
                logger.info("Successfully generated response")
                return generated_text
            else:
                logger.error("No valid response generated - empty candidates or parts")
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = str(e).lower()
            
            if 'api key' in error_msg:
                return "I encountered an authentication error. Please check your API key and try again."
            elif 'quota' in error_msg or 'limit' in error_msg:
                return "I've reached the API usage limit. Please try again later or check your quota."
            elif 'permission' in error_msg:
                return "I don't have permission to access the required API. Please check your API key permissions."
            else:
                return f"I encountered an error while processing your request: {str(e)}"
    
    async def query(self, 
                   user_query: str, 
                   retrieve_docs: bool = True,
                   top_k: int = None,
                   similarity_threshold: float = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response
        
        Args:
            user_query: The user's question
            retrieve_docs: Whether to retrieve documents from the knowledge base
            top_k: Number of documents to retrieve (overrides default)
            similarity_threshold: Similarity threshold for retrieval (overrides default)
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: '{user_query[:50]}...'")
        
        retrieved_docs = []
        context = ""
        
        if retrieve_docs:
            # Retrieve relevant documents
            try:
                logger.info("Retrieving relevant documents...")
                retrieved_docs = await self.retriever(
                    user_query, 
                    k=top_k, 
                    threshold=similarity_threshold
                )
                logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
                
                # Format context
                context = self._format_context(retrieved_docs)
                
            except Exception as e:
                logger.error(f"Error during retrieval: {e}")
                context = "Error accessing knowledge base."
        
        # Create prompt
        prompt = self._create_prompt(user_query, context, self.conversation_history)
        
        # Generate response
        response = await self._generate_response(prompt)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Store in conversation history
        conversation_turn = {
            'timestamp': datetime.now().isoformat(),
            'query': user_query,
            'response': response,
            'retrieved_docs_count': len(retrieved_docs),
            'processing_time': processing_time
        }
        self.conversation_history.append(conversation_turn)
        
        # Keep only last 10 conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        logger.info(f"Generated response in {processing_time:.2f}s")
        
        return {
            'response': response,
            'retrieved_docs': retrieved_docs,
            'retrieved_docs_count': len(retrieved_docs),
            'context_used': context,
            'processing_time': processing_time,
            'timestamp': conversation_turn['timestamp']
        }
    
    async def chat(self):
        """Interactive chat interface"""
        logger.info("Starting interactive chat. Type 'quit', 'exit', or 'bye' to end.")
        logger.info("Type 'clear' to clear conversation history.")
        logger.info("Type 'info' to see collection information.")
        
        print("\n" + "="*60)
        print("🤖 Gemini RAG Agent - Interactive Chat")
        print("="*60)
        print("Ask me anything about airports, facilities, or travel information!")
        print("Commands: 'quit'/'exit'/'bye' to end, 'clear' to reset, 'info' for stats")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Have a great day!")
                    break
                
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🗑️ Conversation history cleared.")
                    continue
                
                elif user_input.lower() == 'info':
                    info = self.embedding_system.get_collection_info()
                    print(f"📊 Knowledge Base Info:")
                    print(f"   Collection: {info.get('collection_name', 'N/A')}")
                    print(f"   Documents: {info.get('document_count', 'N/A')}")
                    print(f"   Model: {info.get('model_name', 'N/A')}")
                    print(f"   Conversations: {len(self.conversation_history)}")
                    continue
                
                # Process the query
                print("🔍 Searching knowledge base...")
                result = await self.query(user_input)
                
                print(f"\n🤖 Assistant: {result['response']}")
                print(f"📚 Sources used: {result['retrieved_docs_count']} documents")
                print(f"⏱️ Response time: {result['processing_time']:.2f}s\n")
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                print(f"❌ Sorry, I encountered an error: {e}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def save_conversation_history(self, filename: str):
        """Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    def load_conversation_history(self, filename: str):
        """Load conversation history from file"""
        try:
            if Path(filename).exists():
                with open(filename, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"Conversation history loaded from {filename}")
            else:
                logger.warning(f"Conversation history file not found: {filename}")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")

async def initialize_agent(api_key: str) -> GeminiRAGAgent:
    """Initialize the RAG agent with knowledge base"""
    logger.info("Initializing Gemini RAG Agent...")
    
    if not api_key:
        logger.error("No API key provided to initialize_agent")
        raise ValueError("API key is required")
    
    try:
        # Initialize embedding system with explicit API key
        logger.info("Creating embedding system...")
        embedding_system = GeminiEmbeddingSystem(
            api_key=api_key,  # Explicitly pass the API key
            chroma_db_path="./chroma_db",
            collection_name="website_embeddings"
        )
        logger.info("Successfully created embedding system")
    except Exception as e:
        logger.error(f"Failed to create embedding system: {e}")
        raise ValueError(f"Failed to create embedding system: {e}")
    
    try:
        # Check if knowledge base exists
        collection_info = embedding_system.get_collection_info()
        document_count = collection_info.get('document_count', 0)
        
        if document_count == 0:
            logger.info("No documents found in knowledge base. Building from cleaned_data...")
            
            directories = ["./cleaned_data"]
            
            # Check if directories exist
            for directory in directories:
                if not Path(directory).exists():
                    logger.error(f"Data directory not found: {directory}")
                    raise ValueError(f"Data directory not found: {directory}")
            
            result = await embedding_system.build_vector_database(directories)
            
            if result['is_built']:
                logger.info(f"Knowledge base built with {result['total_documents']} documents")
            else:
                logger.error("Failed to build knowledge base")
                raise ValueError("Failed to build knowledge base")
        else:
            logger.info(f"Using existing knowledge base with {document_count} documents")
    except Exception as e:
        logger.error(f"Error setting up knowledge base: {e}")
        raise ValueError(f"Error setting up knowledge base: {e}")
    
    try:
        # Create and return agent
        logger.info("Creating RAG agent...")
        agent = GeminiRAGAgent(api_key=api_key, embedding_system=embedding_system)
        logger.info("RAG Agent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create RAG agent: {e}")
        raise ValueError(f"Failed to create RAG agent: {e}")

async def test_agent(api_key: str):
    """Test the RAG agent with sample queries"""
    logger.info("Testing Gemini RAG Agent")
    
    try:
        agent = await initialize_agent(api_key)
        if not agent:
            logger.error("Failed to initialize agent")
            return
    except Exception as e:
        logger.error(f"Failed to initialize agent for testing: {e}")
        return
    
    test_queries = [
        "What facilities are available at Changi Airport?",
        "Tell me about shopping options at the airport",
        "How can I get to Changi Airport from the city?",
        "What entertainment options are available?",
        "Are there any hotels at the airport?",
        "What dining options do you recommend?"
    ]
    
    logger.info("Running test queries:")
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Test Query {i}/{len(test_queries)} ---")
        logger.info(f"Query: {query}")
        
        try:
            result = await agent.query(query)
            
            logger.info(f"Response: {result['response'][:200]}...")
            logger.info(f"Retrieved docs: {result['retrieved_docs_count']}")
            logger.info(f"Processing time: {result['processing_time']:.2f}s")
        except Exception as e:
            logger.error(f"Error processing test query: {e}")
    
    # Save conversation history
    try:
        agent.save_conversation_history("test_conversation_history.json")
        logger.info("Test completed and conversation history saved")
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

async def main():
    """Main function to run the agent"""
    import sys
    
    # Get API key from environment or user input
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
    
    if not api_key:
        print("API key is required!")
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        await test_agent(api_key)
    else:
        # Interactive chat mode
        try:
            agent = await initialize_agent(api_key)
            if agent:
                await agent.chat()
            else:
                logger.error("Failed to initialize agent")
        except Exception as e:
            logger.error(f"Error in main: {e}")
            print(f"Failed to start agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())