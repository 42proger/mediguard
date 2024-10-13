# Guide

## 1. Environment setup

### 1.1. Configure Ollama for inference

First, the OLLAMA_ENDPOINT is defined as the server URL and stored as an environment variable (OLLAMA_HOST). Then, the ChatOllama model is initialized with the "gemma2:27b" LLM, specifying the endpoint, context length (num_ctx=25000), and a temperature. This configuration prepares the system to run inference using the Ollama model for text-based tasks.

```python
import os

OLLAMA_ENDPOINT="http://gpu03:27175"
os.environ['OLLAMA_HOST']=OLLAMA_ENDPOINT
```

```python
from langchain_community.chat_models import ChatOllama

local_llm = "gemma2:27b"
num_ctx=25000
llm = ChatOllama(model=local_llm, base_url=OLLAMA_ENDPOINT, temperature=0, num_ctx=num_ctx)
```

### 1.2. Prompt your local LLM

 The code sends messages to the local model, instructing it to translate English to French as a test. The model is invoked with the prompts, and if configured correctly, it responds in French, displaying the translation and additional text using pretty_print().

```python
from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence. Don't respond in json",
    ),
    ("human", "I love programming. Tell me if you also love programming"),
]
ai_msg = llm.invoke(messages)
ai_msg.pretty_print()
```

If the LLM is properly configured, we will get the following message:

```
================================== Ai Message ==================================

J'adore programmer. Oui, j'aime beaucoup programmer ! C'est très gratifiant de pouvoir créer des choses avec du code. 

What do you like to program?
```

## 2. Solving the problem

The next step involves preparing DataFrames for patient records and LLM interaction checker implamention. More details are provided in the section ["Solving the problem"](solving.md)

## 3. Selecting LLM

In this page will be discribed LLMs and how we chose the model: ["Selecting LLM"](models.md)

## 4. Future Improvements

In this section, we outline potential improvements and new features planned for the future development of our software. ["Future Improvements"](future.md)