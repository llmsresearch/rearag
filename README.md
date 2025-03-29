

<div align="center">
<h1>ReaRAG</h1>
  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)
![GitHub issues](https://img.shields.io/github/issues/llmsresearch/rearag?color=red)
![GitHub stars](https://img.shields.io/github/stars/llmsresearch/rearag?style=social)

**Knowledge-guided Reasoning with Iterative Retrieval Augmented Generation**

</div>

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Extending ReaRAG](#extending-rearag)
- [Differences from Original Paper](#differences-from-the-original-paper)
- [License](#license)
- [Citation](#citation)

---

## Overview

ReaRAG is a factuality-enhanced reasoning model that combines strong reasoning capabilities with retrieval augmentation. It follows the Thought-Action-Observation paradigm:

<div align="center">
  <img src="https://arxiv.org/html/2503.21729v1/x1.png" alt="ReaRAG Workflow" width="60%"/>
</div>

1. **Thought** : The model generates reasoning steps
2. **Action** : The model decides whether to search for more information or finish and provide an answer
3. **Observation** : If a search action is chosen, external knowledge is retrieved to guide further reasoning

This implementation focuses on the algorithm and methodology proposed in the paper, allowing users to use any base LLM for reasoning.

---

## Features

- **Modular Design** - Flexible integration with various LLM providers (OpenAI, Anthropic, Hugging Face, etc.)
- **Multiple Retrieval Sources** - Support for web search, vector databases, and simple in-memory retrieval
- **Configurable Parameters** - Adjust maximum iterations and other settings to your needs
- **Detailed Reasoning Trace** - Full visibility into the reasoning process for explainability
- **Extensible Architecture** - Easy to add new LLM providers and retrievers

---

## Architecture

<div align="center">
  <img src="https://arxiv.org/html/2503.21729v1/x2.png" alt="ReaRAG Architecture" width="60%"/>
</div>

The implementation consists of these main components:

- **Core ReaRAG Engine**: Implements the iterative reasoning algorithm
- **LLM Provider Interface**: Abstract interface for different LLM providers
- **Retriever Interface**: Abstract interface for different retrieval sources
- **Prompt Templates**: Templates for generating prompts for the LLM

---

## Installation

```bash
# Clone the repository
git clone https://github.com/llmsresearch/rearag.git
cd rearag

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Basic Example

```python
from rearag.core.rearag import ReaRAG
from rearag.interfaces.llm_provider import OpenAIProvider
from rearag.interfaces.retriever import SimpleRetriever

# Initialize LLM provider and retriever
llm_provider = OpenAIProvider(model_name="gpt-4-turbo-preview", api_key="YOUR_API_KEY")
retriever = SimpleRetriever(knowledge_base={"who is the prime minister of the uk": "Rishi Sunak is the Prime Minister of the United Kingdom."})

# Initialize ReaRAG
rearag = ReaRAG(
    llm_provider=llm_provider,
    retriever=retriever,
    max_iterations=5,
    verbose=True
)

# Answer a question
result = rearag.answer_question("Who is the current Prime Minister of the UK?")
print(f"Answer: {result['answer']}")

# The reasoning trace is also available
print(result['reasoning_trace'])
```

### Using the Example Script

The repository includes an example script (`example.py`) that demonstrates how to use ReaRAG with different LLM providers and retrievers:

```bash
# Use with default settings (OpenAI LLM and simple retriever)
python example.py --question "What is the capital of France?"

# Use with Anthropic Claude and web search retriever
python example.py --llm anthropic --retriever web --question "Who won the last World Cup?"

# Interactive mode
python example.py --verbose
```

---

## Extending ReaRAG

### Adding a New LLM Provider

1. Create a new class that inherits from `LLMProvider`
2. Implement the `generate` and `get_model_name` methods
3. Register the provider in the `initialize_llm_provider` function in `example.py`

```python
class CustomLLMProvider(LLMProvider):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        # Initialize your custom LLM client
        
    def generate(self, prompt):
        # Implement your generation logic
        return response
        
    def get_model_name(self):
        return self.model_name
```

### Adding a New Retriever

1. Create a new class that inherits from `Retriever`
2. Implement the `retrieve` method
3. Register the retriever in the `initialize_retriever` function in `example.py`

```python
class CustomRetriever(Retriever):
    def __init__(self, **kwargs):
        # Initialize your custom retriever
        
    def retrieve(self, query):
        # Implement your retrieval logic
        return results
```

---

## Differences from the Original Paper

This implementation focuses on the core methodology of the paper without:

1. The fine-tuning process on knowledge-guided reasoning chain data
2. The specific evaluation benchmarks used in the paper (MuSiQue, HotpotQA, IIRC, NQ)
3. The data construction and filtering procedures

The implementation follows the inference-time algorithm described in the paper, allowing users to plug in any base LLM.

---

## License

[MIT License](LICENSE)

---

## Citation

If you use this code for your research, please cite the original paper:

```bibtex
@misc{lee2024rearag,
      title={ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation}, 
      author={Zhicheng Lee and Shulin Cao and Jinxin Liu and Jiajie Zhang and Weichuan Liu and Xiaoyin Che and Lei Hou and Juanzi Li},
      year={2024},
      eprint={2503.21729},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
