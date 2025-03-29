from typing import Dict, List, Optional, Any, Tuple, Union
import json
import logging
from ..interfaces.llm_provider import LLMProvider
from ..interfaces.retriever import Retriever
from ..prompts.templates import get_rearag_prompt
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ReaRAG:
    """
    ReaRAG: Knowledge-guided Reasoning with Iterative Retrieval Augmented Generation
    
    This class implements the core functionality of the ReaRAG model as described in
    the paper: "ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large
    Reasoning Models with Iterative Retrieval Augmented Generation"
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        retriever: Retriever,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the ReaRAG model.
        
        Args:
            llm_provider: The LLM provider interface
            retriever: The retriever interface for external knowledge retrieval
            max_iterations: Maximum number of reasoning iterations allowed
            verbose: Whether to log detailed information
        """
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def _parse_llm_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse the LLM response to extract thought and action.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Tuple of (thought, action) where thought is a string and action is a dictionary
        """
        try:
            # Try to extract thought and action from a structured response
            parts = response.split("Action:")
            if len(parts) < 2:
                logger.warning("Could not parse LLM response for action")
                return response, {"function": "finish", "parameters": {"answer": "I couldn't determine an answer."}}
            
            thought = parts[0].strip()
            action_text = parts[1].strip()
            
            # Parse the action JSON
            # Find the start and end of the JSON object
            json_start = action_text.find("{")
            if json_start == -1:
                # Try to parse as a simple finish or search action
                if "finish" in action_text.lower():
                    # Extract answer text after finish
                    answer_text = action_text.split("finish")[1].strip()
                    return thought, {"function": "finish", "parameters": {"answer": answer_text}}
                elif "search" in action_text.lower():
                    # Extract query text after search
                    query_text = action_text.split("search")[1].strip()
                    return thought, {"function": "search", "parameters": {"query": query_text}}
                else:
                    logger.warning("Could not parse action text")
                    return thought, {"function": "finish", "parameters": {"answer": "I couldn't determine an answer."}}
            
            # Find matching closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(action_text[json_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = json_start + i + 1
                        break
            
            if json_end == -1:
                logger.warning("Could not find matching JSON brace")
                return thought, {"function": "finish", "parameters": {"answer": "I couldn't determine an answer."}}
            
            action_json = action_text[json_start:json_end]
            action = json.loads(action_json)
            
            return thought, action
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response was: {response}")
            return response, {"function": "finish", "parameters": {"answer": "I couldn't determine an answer due to a parsing error."}}
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the ReaRAG methodology.
        
        Args:
            question: The question to answer
            
        Returns:
            A dictionary containing the final answer and the reasoning trace
        """
        # Initialize the reasoning trace
        reasoning_trace = []
        
        # Initialize the context with the question
        context = {
            "question": question,
            "thoughts": [],
            "actions": [],
            "observations": []
        }
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate the prompt for the current state
            prompt = get_rearag_prompt(
                question=question,
                thoughts=context["thoughts"],
                actions=context["actions"],
                observations=context["observations"]
            )
            
            # Get response from LLM
            response = self.llm_provider.generate(prompt)
            
            # Parse the response to get thought and action
            thought, action = self._parse_llm_response(response)
            
            # Add thought to the context
            context["thoughts"].append(thought)
            reasoning_trace.append({"thought": thought})
            
            # Process the action
            context["actions"].append(action)
            reasoning_trace.append({"action": action})
            
            if action["function"] == "search":
                # Execute search and get observation
                query = action["parameters"]["query"]
                if self.verbose:
                    logger.info(f"Searching for: {query}")
                
                observation = self.retriever.retrieve(query)
                context["observations"].append(observation)
                reasoning_trace.append({"observation": observation})
                
                if self.verbose:
                    logger.info(f"Observation: {observation}")
            
            elif action["function"] == "finish":
                # Return the final answer
                answer = action["parameters"]["answer"]
                if self.verbose:
                    logger.info(f"Final answer: {answer}")
                
                return {
                    "answer": answer,
                    "reasoning_trace": reasoning_trace
                }
        
        # If we reach the maximum number of iterations without finishing
        logger.warning("Reached maximum number of iterations without finishing")
        return {
            "answer": "I couldn't determine an answer within the maximum number of iterations.",
            "reasoning_trace": reasoning_trace
        } 