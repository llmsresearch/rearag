#!/usr/bin/env python3
"""
ReaRAG Example Usage

This script demonstrates how to use the ReaRAG system to answer questions
using knowledge-guided reasoning with iterative retrieval.
"""

import os
import argparse
from rearag.core.rearag import ReaRAG
from rearag.interfaces.llm_provider import OpenAIProvider, AnthropicProvider, HuggingFaceProvider
from rearag.interfaces.retriever import SimpleRetriever, WebSearchRetriever, VectorDBRetriever


def setup_sample_knowledge_base():
    """Set up a sample knowledge base for demonstration purposes."""
    knowledge_base = {
        "what is the capital of france": "Paris is the capital of France.",
        "who wrote hamlet": "William Shakespeare wrote Hamlet.",
        "what is the largest planet in our solar system": "Jupiter is the largest planet in our solar system.",
        "what is the boiling point of water": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "who was the first person to walk on the moon": "Neil Armstrong was the first person to walk on the moon on July 20, 1969.",
        "what is the speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "who founded microsoft": "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
        "what is the tallest mountain in the world": "Mount Everest is the tallest mountain in the world, with a height of 8,848.86 meters (29,031.7 feet) above sea level.",
        "what is artificial intelligence": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
        "what is the capital of japan": "Tokyo is the capital of Japan.",
    }
    return knowledge_base


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ReaRAG Example")
    parser.add_argument("--question", type=str, help="Question to answer", required=False)
    parser.add_argument("--llm", type=str, choices=["openai", "anthropic", "huggingface"], default="openai", help="LLM provider to use")
    parser.add_argument("--model", type=str, help="Model name to use with the LLM provider", required=False)
    parser.add_argument("--retriever", type=str, choices=["simple", "web", "vectordb"], default="simple", help="Retriever to use")
    parser.add_argument("--api-key", type=str, help="API key for the LLM provider", required=False)
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of reasoning iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def initialize_llm_provider(args):
    """Initialize the LLM provider based on command line arguments."""
    api_key = args.api_key or os.environ.get(f"{args.llm.upper()}_API_KEY")
    
    if args.llm == "openai":
        model = args.model or "gpt-4-turbo-preview"
        return OpenAIProvider(model_name=model, api_key=api_key)
    
    elif args.llm == "anthropic":
        model = args.model or "claude-3-opus-20240229"
        return AnthropicProvider(model_name=model, api_key=api_key)
    
    elif args.llm == "huggingface":
        model = args.model or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        return HuggingFaceProvider(model_name=model, api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {args.llm}")


def initialize_retriever(args):
    """Initialize the retriever based on command line arguments."""
    if args.retriever == "simple":
        knowledge_base = setup_sample_knowledge_base()
        return SimpleRetriever(knowledge_base=knowledge_base)
    
    elif args.retriever == "web":
        # You would need to provide API keys for these services
        api_key = os.environ.get("SEARCH_API_KEY")
        return WebSearchRetriever(search_provider="google", api_key=api_key)
    
    elif args.retriever == "vectordb":
        # For simplicity, this example uses ChromaDB in-memory
        return VectorDBRetriever(db_type="chromadb", collection_name="rearag_demo")
    
    else:
        raise ValueError(f"Unsupported retriever: {args.retriever}")


def main():
    """Main entry point for the ReaRAG example."""
    args = parse_args()
    
    # Initialize the LLM provider
    llm_provider = initialize_llm_provider(args)
    print(f"Using LLM: {args.llm} with model: {llm_provider.get_model_name()}")
    
    # Initialize the retriever
    retriever = initialize_retriever(args)
    print(f"Using retriever: {args.retriever}")
    
    # Initialize ReaRAG
    rearag = ReaRAG(
        llm_provider=llm_provider,
        retriever=retriever,
        max_iterations=args.max_iterations,
        verbose=args.verbose
    )
    
    # If no question is provided, use an interactive mode
    if not args.question:
        print("\nReaRAG Interactive Mode")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == "exit":
                break
            
            print("\nThinking...\n")
            result = rearag.answer_question(question)
            
            print(f"\nAnswer: {result['answer']}")
            
            if args.verbose:
                print("\nReasoning trace:")
                for i, item in enumerate(result['reasoning_trace']):
                    key = list(item.keys())[0]
                    print(f"{key.capitalize()}: {item[key]}")
    else:
        # Answer a single question
        print(f"\nQuestion: {args.question}")
        print("\nThinking...\n")
        
        result = rearag.answer_question(args.question)
        
        print(f"\nAnswer: {result['answer']}")
        
        if args.verbose:
            print("\nReasoning trace:")
            for i, item in enumerate(result['reasoning_trace']):
                key = list(item.keys())[0]
                print(f"{key.capitalize()}: {item[key]}")


if __name__ == "__main__":
    main()