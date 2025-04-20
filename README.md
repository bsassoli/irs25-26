# RAG Markdown Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on knowledge stored in a Markdown file. This implementation uses LlamaIndex as the framework, ChromaDB for vector storage, and OpenAI's GPT-4o for language generation.

## Features

- Processes and indexes Markdown text using a custom `HybridMarkdownSentenceParser`
- Uses ChromaDB as a persistent vector store for efficient retrieval
- Leverages OpenAI's GPT-4o for high-quality responses
- Provides source citations with relevance scores
- Includes both CLI and Streamlit web interfaces

## Project Structure

```
rag-chatbot/
├── app.py                     # Streamlit web application
├── rag_chatbot.py             # Core RAG chatbot implementation
├── data/
│   └── text.md                # Knowledge base in Markdown format
├── src/
│   └── hybrid_parser.py       # Custom Markdown parser
├── chroma_db/                 # Persistent vector database (created on first run)
├── Dockerfile                 # Docker configuration
└── requirements.txt           # Python dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Docker (optional)

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

### Option 2: Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t rag-chatbot .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 -e OPENAI_API_KEY=your-api-key-here rag-chatbot
   ```

## Usage

### Command Line Interface

Run the chatbot in CLI mode:

```bash
python rag_chatbot.py
```

### Web Interface

Start the Streamlit web application:

```bash
streamlit run app.py
```

The web interface will be available at http://localhost:8501

## Configuration Options

The chatbot can be configured with the following parameters:

- **OpenAI API Key**: Required for accessing the OpenAI models
- **Model Name**: GPT-4o by default, can be changed to other OpenAI models
- **Knowledge Base Path**: Path to your Markdown file (default: `data/text.md`)
- **Embedding Model**: OpenAI embedding model to use (default: `text-embedding-3-large`)
- **Chunk Size**: Maximum token size for text chunks (default: 512)
- **Chunk Overlap**: Overlap between chunks to maintain context (default: 50)
- **Number of Chunks to Retrieve**: How many relevant chunks to retrieve (default: 3)

## Customizing the Knowledge Base

Replace the `data/text.md` file with your own Markdown content. The system uses the custom `HybridMarkdownSentenceParser` from `src/hybrid_parser.py` to process the Markdown structure while maintaining the hierarchical context.

## License

[MIT License](LICENSE)