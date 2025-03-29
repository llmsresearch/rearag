from typing import List, Dict, Any, Optional

REARAG_BASE_PROMPT = """
You are ReaRAG, a knowledge-guided reasoning model that can search for information to answer questions factually.

You need to solve the following question by thinking step by step. You will follow the Thought-Action-Observation paradigm:

1. First, you'll provide a Thought: This is your internal reasoning process where you think about what you know and what information you need.

2. Then, you'll choose one of the following actions:
   - Search: Look up information using a search query
   - Finish: Complete the task and provide a final answer

3. If you choose Search, you'll receive an Observation, which is the result of your search. You can then continue with another Thought.

Always use the following format:

Thought: <your step-by-step reasoning>
Action: {"function": "search", "parameters": {"query": "your search query here"}}
OR
Action: {"function": "finish", "parameters": {"answer": "your final answer here"}}

Remember:
- You can perform multiple search actions as needed
- Search for only one thing at a time with specific queries
- Use the information from observations to inform your next thought
- When you have enough information, use the finish action
- Always provide your final answer in a clear, concise way that directly addresses the question
- Don't exceed {max_iterations} iterations
"""

REARAG_SESSION_PROMPT = """
Question: {question}

{thought_action_observation_history}

Thought:
"""

def get_rearag_prompt(
    question: str,
    thoughts: List[str] = None,
    actions: List[Dict[str, Any]] = None,
    observations: List[str] = None,
    max_iterations: int = 10
) -> str:
    """
    Generate the prompt for the ReaRAG model.
    
    Args:
        question: The question to answer
        thoughts: List of previous thoughts
        actions: List of previous actions
        observations: List of previous observations
        max_iterations: Maximum number of iterations allowed
        
    Returns:
        The formatted prompt
    """
    thoughts = thoughts or []
    actions = actions or []
    observations = observations or []
    
    # Create base prompt with max_iterations
    base_prompt = REARAG_BASE_PROMPT.format(max_iterations=max_iterations)
    
    # Build history of thought-action-observation
    history = ""
    for i in range(len(thoughts)):
        history += f"Thought: {thoughts[i]}\n"
        
        if i < len(actions):
            action = actions[i]
            if action["function"] == "search":
                history += f"Action: {{'function': 'search', 'parameters': {{'query': '{action['parameters']['query']}'}}}}\n"
            elif action["function"] == "finish":
                history += f"Action: {{'function': 'finish', 'parameters': {{'answer': '{action['parameters']['answer']}'}}}}\n"
            
            if i < len(observations) and action["function"] == "search":
                history += f"Observation: {observations[i]}\n\n"
    
    # Combine base prompt and session prompt
    full_prompt = base_prompt + "\n\n" + REARAG_SESSION_PROMPT.format(
        question=question,
        thought_action_observation_history=history
    )
    
    return full_prompt 