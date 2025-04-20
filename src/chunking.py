from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
import pandas as pd
from hybrid_parser import HybridMarkdownSentenceParser
import chromadb
from test_questions import TEST_QUESTIONS as test_questions, TEST_ANSWERS as test_answers
from prompt import PROMPT
from app import parse_and_embed_kb, ChatBot  # Add import for referenced functions

def test_chunking_strategies(chroma_client):
    """
    Test different chunking strategies to find the optimal configuration.
    """
    # Configure base settings
    Settings.llm = OpenAI(model="gpt-4o", temperature=0)
    Settings.embed_model = OpenAIEmbedding()
    
    # Load documents
    documents = SimpleDirectoryReader(
        input_dir="./data/chapters",
        recursive=True
    ).load_data()
    
    # Define chunking strategies to test
    chunking_strategies = {
        "size256ol20": HybridMarkdownSentenceParser(chunk_size=256, chunk_overlap=20),
        "size512ol20": HybridMarkdownSentenceParser(chunk_size=512, chunk_overlap=20),
        "size1024ol20": HybridMarkdownSentenceParser(chunk_size=1024, chunk_overlap=20),
        "size256ol50": HybridMarkdownSentenceParser(chunk_size=256, chunk_overlap=50),
        "size512ol50": HybridMarkdownSentenceParser(chunk_size=512, chunk_overlap=50),
        "size1024ol50": HybridMarkdownSentenceParser(chunk_size=1024, chunk_overlap=50),
        "size256ol100": HybridMarkdownSentenceParser(chunk_size=256, chunk_overlap=100),
        "size512ol100": HybridMarkdownSentenceParser(chunk_size=512, chunk_overlap=100),
        "size1024ol100": HybridMarkdownSentenceParser(chunk_size=1024, chunk_overlap=100),
    }
    
    # Create a standard prompt template for all tests
    qa_template = PromptTemplate(PROMPT)
    
    # Initialize evaluators
    llm = OpenAI(model="gpt-4", temperature=0)
    from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator, CorrectnessEvaluator
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)
    correctness_evaluator = CorrectnessEvaluator(llm=llm)
    
    # Results container
    all_results = []
    
    # Test all chunking strategies
    for chunk_name, chunker in chunking_strategies.items():
        print(f"\nTesting chunking strategy: {chunk_name}")
        
        # Create a new collection for this test
        collection_name = f"test_{chunk_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        chroma_collection = chroma_client.create_collection(collection_name)
        
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.storage.storage_context import StorageContext
        
        # Set up vector store with this chunking strategy
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Parse documents with this chunker
        Settings.text_parser = chunker
        
        # Create index with this chunking strategy
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Create query engine with response synthesizer
        query_engine = index.as_query_engine(
            response_synthesizer=get_response_synthesizer(
                response_mode="tree_summarize",
                text_qa_template=qa_template
            )
        )
        
        # Process all questions for this chunking strategy
        config_results = []
        for i, question in enumerate(test_questions):
            print(f"  Testing question: {question[:30]}...")
            
            # Get response
            response = query_engine.query(question)
            response_text = str(response)
            
            # Get source nodes
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            context_strings = [node.node.text for node in source_nodes] if source_nodes else []
            
            # Record results
            result = {
                "chunking_strategy": chunk_name,
                "question": question,
                "response": response_text,
                "num_source_nodes": len(source_nodes),
                "avg_chunk_size": sum(len(context) for context in context_strings) / len(context_strings) if context_strings else 0
            }
            
            # Evaluate faithfulness
            if context_strings:
                faithfulness_result = faithfulness_evaluator.evaluate(
                    query=question,
                    response=response_text,
                    contexts=context_strings
                )
                result["faithfulness_score"] = faithfulness_result.score
                result["faithfulness_feedback"] = faithfulness_result.feedback
            else:
                result["faithfulness_score"] = None
                result["faithfulness_feedback"] = "No source nodes available for evaluation"
            
            # Evaluate relevancy
            relevancy_result = relevancy_evaluator.evaluate(
                query=question,
                response=response_text
            )
            result["relevancy_score"] = relevancy_result.score
            result["relevancy_feedback"] = relevancy_result.feedback
            
            # Evaluate correctness using the provided reference answers
            correctness_result = correctness_evaluator.evaluate(
                query=question,
                response=response_text,
                reference=test_answers[i]
            )
            result["correctness_score"] = correctness_result.score
            result["correctness_feedback"] = correctness_result.feedback
            
            config_results.append(result)
            all_results.append(result)
        
        # Calculate and display average scores for this configuration after all questions
        if any(r["faithfulness_score"] is not None for r in config_results):
            faith_scores = [r["faithfulness_score"] for r in config_results if r["faithfulness_score"] is not None]
            avg_faith = sum(faith_scores) / len(faith_scores)
        else:
            avg_faith = 0
            
        avg_relevance = sum(r["relevancy_score"] for r in config_results) / len(config_results)
        avg_correctness = sum(r["correctness_score"] for r in config_results) / len(config_results)
        
        print(f"  Chunking strategy {chunk_name} results:")
        print(f"    Avg Faithfulness: {avg_faith:.2f}, Avg Relevance: {avg_relevance:.2f}, Avg Correctness: {avg_correctness:.2f}")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Aggregate results by chunking strategy
    agg_results = results_df.groupby(['chunking_strategy']).agg({
        'faithfulness_score': ['mean', 'std'],
        'relevancy_score': ['mean', 'std'],
        'correctness_score': ['mean', 'std'],
        'num_source_nodes': 'mean',
        'avg_chunk_size': 'mean'
    }).reset_index()
    
    # Find the best configuration - now considering all three metrics
    # You could weight these metrics differently if desired
    best_combined = agg_results.sort_values(
        by=[('correctness_score', 'mean'), ('faithfulness_score', 'mean'), ('relevancy_score', 'mean')], 
        ascending=False
    ).iloc[0]
    
    print("\nBest configuration:")
    print(f"Chunking strategy: {best_combined['chunking_strategy']}")
    print(f"Avg Faithfulness: {best_combined[('faithfulness_score', 'mean')]:.2f}")
    print(f"Avg Relevance: {best_combined[('relevancy_score', 'mean')]:.2f}")
    print(f"Avg Correctness: {best_combined[('correctness_score', 'mean')]:.2f}")
    
    # Save detailed results
    results_df.to_csv("chunking_test_results.csv", index=False)
    agg_results.to_csv("chunking_agg_results.csv")
    
    return results_df, agg_results, best_combined

# Example usage
if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient("./chroma_db")
    results_df, agg_results, best_config = test_chunking_strategies(chroma_client)
    
    # Use the best configuration - extract chunk size and overlap from the name
    best_strategy = best_config[('chunking_strategy', '')] if isinstance(best_config['chunking_strategy'], pd.Series) else best_config['chunking_strategy']
    
    # Alternative approach if the above still gives an error
    # best_strategy = agg_results.iloc[0]['chunking_strategy']
    
    size_part = best_strategy.split('ol')[0].replace('size', '')
    overlap_part = best_strategy.split('ol')[1]
    
    best_chunk_size = int(size_part)
    best_chunk_overlap = int(overlap_part)
    
    print(f"Best chunking parameters: size={best_chunk_size}, overlap={best_chunk_overlap}")
    Settings.text_parser = HybridMarkdownSentenceParser(chunk_size=best_chunk_size, chunk_overlap=best_chunk_overlap)
    # Re-run the embedding and indexing process with the best parameters