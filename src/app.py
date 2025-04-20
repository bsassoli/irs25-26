import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from hybrid_parser import HybridMarkdownSentenceParser
from prompt import PROMPT



class ChatBot:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.chat_history = []
        
        # Create a custom prompt that emphasizes philosophical expertise
        self.qa_template = PromptTemplate(PROMPT)    
        
        # Create query engine with custom response synthesizer
        self.query_engine = self.index.as_query_engine(
                response_synthesizer=get_response_synthesizer(
                response_mode="tree_summarize",
                text_qa_template=self.qa_template
            )
        )
    
    def ask(self, question: str) -> str:
        """Process a question and return a response based on your knowledge."""
        response = self.query_engine.query(question)
        self.chat_history.append({"question": question, "answer": str(response)})
        return str(response)

def parse_and_embed_kb(chroma_client) -> VectorStoreIndex:
    # Configure settings
    Settings.llm = OpenAI(model="gpt-4", temperature=0)
    Settings.embed_model = OpenAIEmbedding()
    Settings.text_parser = HybridMarkdownSentenceParser()

    # Collection name for our knowledge base
    collection_name = "philosophy_knowledge"
    
    # Check if collection exists
    collections = chroma_client.list_collections()
    collection_exists = any(collection.name == collection_name for collection in collections)
    
    # Create or get the collection
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Only load and index documents if the collection doesn't exist or is empty
    if not collection_exists or chroma_collection.count() == 0:
        print("Creating new knowledge base index...")
        
        # Load documents from knowledge base directory
        documents = SimpleDirectoryReader(
            input_dir="./data/chapters",
            recursive=True
        ).load_data()

        # Create index with ChromaDB storage
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("Knowledge base successfully indexed.")
    else:
        print("Loading existing knowledge base...")
        # Load the existing index from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
        print("Knowledge base loaded.")
    
    return index


def chat_with_kb():
    """Interactive chat loop with the knowledge base."""
    
      # Initialize the ChromaDB client
    chroma_client = chromadb.PersistentClient("./chroma_db")
    
    # Initialize the index and chatbot
    index = parse_and_embed_kb(chroma_client)
    chatbot = ChatBot(index)
    
    print("Welcome to the Recipes for Science Chatbot! Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
            
        # Get and print response
        if user_input:
            try:
                response = chatbot.ask(user_input)
                print("\nBot:", response)
            except Exception as e:
                print("\nError:", str(e))
                print("Please try asking your question in a different way.")

if __name__ == "__main__":
    # Initialize the ChromaDB client
    chroma_client = chromadb.PersistentClient("./data/chroma_db")
    
    # Start the chat loop
    chat_with_kb()
