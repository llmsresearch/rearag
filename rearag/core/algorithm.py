"""
This module contains the algorithms described in the ReaRAG paper, implemented as
Python pseudocode functions.

Note that Algorithm 1 (data construction) is primarily for reference since we're 
not implementing the data construction and fine-tuning parts in this repository.

Algorithm 2 (inference) is implemented in the ReaRAG class, but it's reproduced
here as a standalone function for clarity.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable


def algorithm1_data_construction(
    Q: str,
    LRM: Callable[[str], str],
    RAG_engine: Callable[[str], str],
    max_length: int = 10
) -> List[Dict[str, Any]]:
    """
    Algorithm 1 from the ReaRAG paper: Automatic Data Construction
    
    This function is an implementation of the data construction algorithm described
    in the paper. It's meant as a reference and not used in the actual code.
    
    Args:
        Q: The question to generate data for
        LRM: A function that simulates a Large Reasoning Model
        RAG_engine: A function that simulates the RAG engine
        max_length: Maximum length of the reasoning chain
        
    Returns:
        A list of thought-action-observation triples
    """
    # Define M_1 to M_k
    M_1 = "Can you solve the given question by searching for information? " + \
          "Think about what you know and what you need to search for." + \
          f"Question: {Q}"
    
    M_2 = "Based on your thought, choose an action from {Search, Finish}. " + \
          "If you need more information, choose Search; otherwise, choose Finish."
    
    M_3 = "If you chose Search, provide a query for the search engine."
    
    # Initial state
    t = []  # List of thought-action-observation triples
    i = 0
    
    # Main loop
    while i < max_length:
        # Generate thought
        if i == 0:
            Thought_i = LRM(M_1)
        else:
            context = "\n".join([f"Thought {j+1}: {t[j]['thought']}\n" + 
                               f"Action {j+1}: {t[j]['action']}\n" + 
                               (f"Observation {j+1}: {t[j]['observation']}\n" if t[j]['action'] == 'Search' else "") 
                               for j in range(i)])
            Thought_i = LRM(f"Given your previous thoughts and the information obtained, " +
                          f"what's your next thought on solving the question: '{Q}'?\n\n" +
                          f"Previous steps:\n{context}")
        
        # Generate action
        context = "\n".join([f"Thought {j+1}: {t[j]['thought']}\n" + 
                           f"Action {j+1}: {t[j]['action']}\n" + 
                           (f"Observation {j+1}: {t[j]['observation']}\n" if t[j]['action'] == 'Search' else "") 
                           for j in range(i)])
        context += f"Thought {i+1}: {Thought_i}"
        Action_i = LRM(f"{M_2}\n\nPrevious steps:\n{context}")
        
        # Process the action
        if Action_i.strip() == "Search":
            # Generate query
            context = "\n".join([f"Thought {j+1}: {t[j]['thought']}\n" + 
                               f"Action {j+1}: {t[j]['action']}\n" + 
                               (f"Observation {j+1}: {t[j]['observation']}\n" if t[j]['action'] == 'Search' else "") 
                               for j in range(i)])
            context += f"Thought {i+1}: {Thought_i}\nAction {i+1}: Search"
            Query_i = LRM(f"{M_3}\n\nPrevious steps:\n{context}")
            
            # Get observation from RAG engine
            Observation_i = RAG_engine(Query_i)
            
            # Add to trace
            t.append({
                "thought": Thought_i,
                "action": "Search",
                "query": Query_i,
                "observation": Observation_i
            })
        else:  # Action_i == "Finish"
            # Generate answer
            context = "\n".join([f"Thought {j+1}: {t[j]['thought']}\n" + 
                               f"Action {j+1}: {t[j]['action']}\n" + 
                               (f"Observation {j+1}: {t[j]['observation']}\n" if t[j]['action'] == 'Search' else "") 
                               for j in range(i)])
            context += f"Thought {i+1}: {Thought_i}\nAction {i+1}: Finish"
            Answer_i = LRM(f"Based on your thoughts and the information gathered, " + 
                         f"what is the final answer to the question: '{Q}'?\n\n" +
                         f"Previous steps:\n{context}")
            
            # Add to trace
            t.append({
                "thought": Thought_i,
                "action": "Finish",
                "answer": Answer_i
            })
            
            # Exit loop early if action is Finish
            break
        
        i += 1
    
    return t


def algorithm2_inference(
    Q: str,
    ReaRAG_model: Callable[[str], Dict[str, Any]],
    RAG_engine: Callable[[str], str],
    max_iterations: int = 10
) -> str:
    """
    Algorithm 2 from the ReaRAG paper: Inference
    
    This is the inference algorithm for ReaRAG. It's a standalone implementation
    of what the ReaRAG class does.
    
    Args:
        Q: The question to answer
        ReaRAG_model: A function that simulates the fine-tuned ReaRAG model
        RAG_engine: A function that simulates the RAG engine
        max_iterations: Maximum number of iterations
        
    Returns:
        The final answer to the question
    """
    # Initialize trace
    trace = []
    
    # Main loop
    for i in range(max_iterations):
        # Build context from trace
        context = "\n".join([f"Thought {j+1}: {trace[j]['thought']}\n" + 
                           f"Action {j+1}: {trace[j]['action']['function']}\n" + 
                           (f"Observation {j+1}: {trace[j]['observation']}\n" if trace[j]['action']['function'] == 'search' else "") 
                           for j in range(len(trace))])
        
        # Generate thought and action using ReaRAG model
        response = ReaRAG_model(f"Question: {Q}\n\n" + 
                              (f"Previous steps:\n{context}\n\n" if context else ""))
        
        # Extract thought and action
        thought = response.get("thought", "")
        action = response.get("action", {})
        
        # Add thought and action to trace
        trace_item = {
            "thought": thought,
            "action": action
        }
        
        # Process the action
        if action.get("function") == "search":
            # Extract query
            query = action.get("parameters", {}).get("query", "")
            
            # Get observation from RAG engine
            observation = RAG_engine(query)
            
            # Add observation to trace
            trace_item["observation"] = observation
        
        # Add to trace
        trace.append(trace_item)
        
        # Check if action is finish
        if action.get("function") == "finish":
            # Extract answer
            answer = action.get("parameters", {}).get("answer", "")
            return answer
    
    # If we reach the maximum number of iterations without finishing
    return "I couldn't determine an answer within the maximum number of iterations." 