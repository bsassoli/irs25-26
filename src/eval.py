import chromadb
from app import ChatBot, chat_with_kb, parse_and_embed_kb


from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator, 
    CorrectnessEvaluator
)
from llama_index.llms.openai import OpenAI
from typing import List, Dict, Any
import json
import pandas as pd
from datetime import datetime
from test_questions import TEST_QUESTIONS, TEST_ANSWERS

def run_evaluations(chatbot, test_questions: List[str], expected_answers: List[str] = None):
    """
    Run a suite of evaluations on the chatbot.
    
    Args:
        chatbot: The ChatBot instance to evaluate
        test_questions: List of test questions
        expected_answers: Optional list of ground truth answers for correctness evaluation
    
    Returns:
        Dictionary containing evaluation results
    """
    # Initialize evaluators
    llm = OpenAI(model="gpt-4", temperature=0)
    
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)
    
    # Only initialize correctness evaluator if expected answers are provided
    correctness_evaluator = None
    if expected_answers and len(expected_answers) == len(test_questions):
        correctness_evaluator = CorrectnessEvaluator(llm=llm)
    
    results = []
    
    # Run evaluations for each question
    for i, question in enumerate(test_questions):
        print(f"Evaluating question {i+1}/{len(test_questions)}: {question}")
        
        # Get response directly from the query engine to access source nodes
        response = chatbot.query_engine.query(question)
        response_text = str(response)
        
        # Get source nodes from the response object
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        # Extract context strings from nodes
        context_strings = [node.node.text for node in source_nodes] if source_nodes else []
        
        # Run evaluations
        eval_result = {
            "question": question,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # Faithfulness - evaluates if the answer is faithful to the source documents
        if context_strings:
            faithfulness_result = faithfulness_evaluator.evaluate(
                query=question,
                response=response_text,
                contexts=context_strings
            )
            eval_result["faithfulness_score"] = faithfulness_result.score
            eval_result["faithfulness_feedback"] = faithfulness_result.feedback
        else:
            eval_result["faithfulness_score"] = None
            eval_result["faithfulness_feedback"] = "No source nodes available for evaluation"
        
        # Relevancy - evaluates if the answer is relevant to the question
        relevancy_result = relevancy_evaluator.evaluate(
            query=question,
            response=response_text
        )
        eval_result["relevancy_score"] = relevancy_result.score
        eval_result["relevancy_feedback"] = relevancy_result.feedback
        
        # Correctness - evaluates if the answer is correct compared to ground truth
        if correctness_evaluator and expected_answers:
            correctness_result = correctness_evaluator.evaluate(
                query=question,
                response=response_text,
                reference=expected_answers[i]
            )
            eval_result["correctness_score"] = correctness_result.score
            eval_result["correctness_feedback"] = correctness_result.feedback
        else:
            eval_result["correctness_score"] = None
            eval_result["correctness_feedback"] = "No ground truth provided"
        
        # Update chatbot history
        chatbot.chat_history.append({"question": question, "answer": response_text})
        
        results.append(eval_result)
    
    return results

def analyze_evaluation_results(results: List[Dict[str, Any]]):
    """
    Analyze and summarize evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
    
    Returns:
        Summary statistics and detailed results
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        "num_questions": len(df),
        "avg_faithfulness": df["faithfulness_score"].mean() if "faithfulness_score" in df else None,
        "avg_relevancy": df["relevancy_score"].mean() if "relevancy_score" in df else None,
        "avg_correctness": df["correctness_score"].mean() if "correctness_score" in df else None,
    }
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total questions evaluated: {summary['num_questions']}")
    print(f"Average faithfulness score: {summary['avg_faithfulness']:.2f}" if summary['avg_faithfulness'] else "Faithfulness not evaluated")
    print(f"Average relevancy score: {summary['avg_relevancy']:.2f}" if summary['avg_relevancy'] else "Relevancy not evaluated")
    print(f"Average correctness score: {summary['avg_correctness']:.2f}" if summary['avg_correctness'] else "Correctness not evaluated")
    
    return summary, df

def save_evaluation_results(results, output_file="evaluation_results.json"):
    """Save evaluation results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def run_philosophy_chatbot_eval(chatbot):
    """Run a comprehensive evaluation of the philosophy chatbot."""

    
    # Run evaluations
    results = run_evaluations(chatbot, TEST_QUESTIONS, TEST_ANSWERS)
    
    # Analyze results
    summary, results_df = analyze_evaluation_results(results)
    
    # Save results
    save_evaluation_results(results)
    
    return summary, results_df

# Example usage (add to your main code):
if __name__ == "__main__":
    # Initialize the ChromaDB client
    chroma_client = chromadb.PersistentClient("./data/chroma_db")
    
    # Initialize the index and chatbot
    index = parse_and_embed_kb(chroma_client)
    chatbot = ChatBot(index)
    
    # Run evaluations
    print("Starting chatbot evaluation...")
    summary, results_df = run_philosophy_chatbot_eval(chatbot)
    
    # After evaluation, you can start the regular chat loop
    chat_with_kb()