# About

> The project was developed as part of the EuroCC Hackathon.

This module reads patient and medicine data, extracts relevant medical information (such as drug names, dosages, allergies, and illnesses), performs language model inference using Ollama configured through `LangChain`, and integrates RAG (Retrieval-Augmented Generation) for handling unstructured data like PDF documents. The goal is to provide AI-driven insights into patient medication histories and drug prescriptions with a particular focus on managing polypharmacy in hospitalized patients.

## Ollama Configuration for Inference

This section sets up the Ollama inference model for local language processing using `LangChain`. The model is specified by the local_llm variable and is used for inference, with contextual token limits set by num_ctx.

Parameters:

- `OLLAMA_ENDPOINT` (str): URL where the Ollama inference model is hosted.
- `local_llm` (str): Model ID for the local language model to be used (in this case, "gemma2:27b").
- `num_ctx` (int): Context length for the inference model.
- `temperature` (float): Controls the randomness of the model's responses.

## Patient Data Extraction from Excel Files

This part of the code processes patient data stored across multiple sheets in an Excel file. It goes through each sheet that contains patient details, retrieves key medical information like drug names, dosages, and allergies, and then organizes this data into a table-like structure. Each patient's information is captured and formatted into rows for easy access, with patient IDs used to categorize the records. The resulting organized data is then stored for further analysis or processing. This ensures that the raw medical data is clean, organized, and easy to work with.

## Integrating RAG (Retrieval-Augmented Generation)

This code demonstrates the process of implementing Retrieval-Augmented Generation (RAG) by loading patient-related documents in PDF format. RAG is useful for integrating information retrieval (from sources like PDFs) into the language model's inference process, helping to generate more accurate and data-informed results.

In this context, `PyPDFLoader` from `LangChain` is used to load and process medical documents, which can be fed into the language model for augmenting inference with relevant real-world data (e.g., medical reports, prescription histories).


### Dependencies

- `pandas` for data handling.
- `LangChain` for model configuration and inference.
- `Ollama` for local inference processing.
- `tqdm` for progress tracking.
- `PyPDFLoader` for PDF document processing.