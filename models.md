# Selecting LLM

## About models

Large Language Models (LLMs) have revolutionized the field of natural language processing (NLP) by enabling machines to understand and generate human-like text. For MediGuard, LLMs are essential for processing complex medication-related information, such as drug interactions, side effects, and patient-specific variables. These models can analyze vast amounts of unstructured text, such as medical documentation and drug information, and extract relevant insights that would be difficult to capture using traditional algorithms.

### gemma2:27b (Google)
**27 billion** parameters; size is **25000 tokens** ; optimized for general language tasks with a transformer-based architecture.

### llama3.1:latest (Meta)
Parameter count unspecified; size is **8192 tokens**; specialized for maintaining context in multi-turn conversations.

### phi3.5:3.8b-mini-instruct-fp16 (Microsoft)
**3.8 billion** parameters; size is **4096 tokens**; uses FP16 for efficiency; focused on instruction-following tasks.

### nemotron-mini:4b-instruct-fp16 (Nvidia)
**4 billion parameters**; size is **2048 tokens**; also uses FP16; optimized for executing user instructions in resource-constrained environments.


## Model selecting


We created an evaluation framework where each model is tested with representative inputs relevant to our use case, measuring response quality and performance. 


Our team compared four available models — `Gemma:27b`, `Llama3.1`, `Nemotron-mini:4b-instruct-fp16`, and `Phi3.5:3.8b-mini-instruct-fp16` based on their performance in extracting, processing, and analyzing medication information from PDF files.


```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the models and their respective configurations
models = {
    "gemma2:27b",
    "llama3.1:latest",
    "nemotron-mini:4b-instruct-fp16", 
    "phi3.5:3.8b-mini-instruct-fp16",
}

# Sample inputs relevant to your use case
sample_inputs = [
    "Can I take Drug A with Drug B?",
    "What are the side effects of Drug C?",
    "Is Drug D suitable for patients with condition E?",
]

# Initialize results storage
results = {}

# Evaluate each model
for model_name, model_path in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    total_time = 0
    responses = []
    
    for input_text in sample_inputs:
        start_time = time.time()
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        
        response_time = end_time - start_time
        total_time += response_time
        responses.append((input_text, response))
    
    average_time = total_time / len(sample_inputs)
    results[model_name] = {
        "average_time": average_time,
        "responses": responses,
    }

# Analyze results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Average Response Time: {metrics['average_time']:.2f} seconds")
    for input_text, response in metrics["responses"]:
        print(f"Input: {input_text}\nResponse: {response}\n")
```


To ensure a fair comparison, we tested each model using the same code, changing only the model name. `Gemma:27b` outperformed the others, handling the task of extracting and organizing information into JSON format more efficiently and without significant issues. The output was consistent with our expectations, and the overall process was smoother compared to the results from other models.

When we tested the others, we encountered various problems. These models produced incomplete or less reliable outputs, leading to inconsistencies during the analysis, which made them less suitable for our needs.

llama mistakes
```json
{
    "Bisoprolol": {
        "non_json_content": "Here are the extracted information about drug interactions, effects of the medicine at different doses, effects on different age groups (children, adults, elderly), warnings or special considerations for people with allergies, potential adverse effects or warnings:\n\n**Drug Interactions:**\n\n* Medicines for controlling blood pressure or heart problems (e.g. amiodarone, amlodipine, clonidine)\n* Medicines for depression (e.g. imipramine, amitriptyline, moclobemide)\n* Medicines to treat mental illness (e.g. 
        
        ...

        \n* Driving and using machines: The use of bisoprolol may sometimes result in dizziness or fatigue. If you suffer from these side effects, do not operate vehicles and/or machines."
    },
}

```

Based on these observations, we selected 
`Gemma:27b` for its superior performance in this specific task.
