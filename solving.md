# Solving the problem

## 1. Data Preparation

### 1.1. Patient Cards

In this section, patient data is processed from Excel sheets starting with "PATIENT," extracting age, medication details, allergies, and illnesses. Unique entries are stored in sets and organized into a list of dictionaries, which is then converted into a DataFrame with the patient ID as an index.

```python
import pandas as pd

# Load the Excel file
file_path = 'hackathon.xlsx'
excel_data = pd.ExcelFile(file_path)

patient_data = []

# Iterate over each sheet in the Excel file
for sheet_name in excel_data.sheet_names:
    if sheet_name.startswith("PATIENT"):
        # Read the patient sheet
        patient_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Extract the patient's age
        patient_age = patient_df.at[0, 1]

        # Initialize sets to store unique allergies and illnesses
        patient_allergies = set()
        patient_illnesses = set()

        # Iterate over rows to extract multiple entries
        row = 2
        while row < len(patient_df) and pd.notna(patient_df.at[row, 0]):  # Continue until we find an empty cell in the drug name column
            drug_name = patient_df.at[row, 0]
            dosage = patient_df.at[row, 1]
            frequency = patient_df.at[row, 2]
            route = patient_df.at[row, 3]
            allergies = patient_df.at[row, 7]
            illnesses = patient_df.at[row, 9]

            if pd.notna(allergies):
                patient_allergies.add(allergies)
            if pd.notna(illnesses):
                patient_illnesses.add(illnesses)
            
            # Prepare patient info
            patient_info = {
                'Patient_ID': sheet_name.split()[-1],
                'Age': patient_age,
                'Drug_Name': drug_name,
                'Dosage': dosage,
                'Frequency': frequency,
                'Route': route
            }
            patient_data.append(patient_info)
            row += 1

        # Update the entries in patient_data with allergies and illnesses
        for entry in patient_data:
            if entry['Patient_ID'] == sheet_name.split()[-1]:
                entry['Allergies'] = ', '.join(patient_allergies) if patient_allergies else 'None'
                entry['Illnesses'] = ', '.join(patient_illnesses) if patient_illnesses else 'None'

# Convert the list of dictionaries to a DataFrame
all_patients_df = pd.DataFrame(patient_data)

# Set index and display the DataFrame
all_patients_df.set_index(['Patient_ID'], inplace=True)
all_patients_df.head(10)
```
The patient DataFrame looks like this:

| Patient_ID | Age  | Drug_Name                   | Dosage  | Frequency | Route  | Allergies                 | Illnesses                           |
|------------|------|-----------------------------|---------|-----------|--------|---------------------------|-------------------------------------|
| 1          | 84.0 | Levetiracetam 500mg        | 1500mg  | b.d.      | orally | lactose, Penicillin       | Dementia, Lactose intolerant, Epilepsy |
| 1          | 84.0 | Carbamazepine 200mg        | 400mg   | b.d.      | orally | lactose, Penicillin       | Dementia, Lactose intolerant, Epilepsy |
| 1          | 84.0 | Amitriptilline 10mg        | 10mg    | Morning   | orally | lactose, Penicillin       | Dementia, Lactose intolerant, Epilepsy |
| 1          | 84.0 | Topamax (topiramate)       | 150mg   | b.d.      | orally | lactose, Penicillin       | Dementia, Lactose intolerant, Epilepsy |
| 1          | 84.0 | Paracetamol 1gr            | 1000mg  | prn       | orally | lactose, Penicillin       | Dementia, Lactose intolerant, Epilepsy |


### 1.2. Medicine Files

For the medication data, it needs to deal with a large set of PDFs containing detailed information on various drugs. These PDFs are processed using Retrieval-Augmented Generation (RAG), an approach that combines document retrieval with natural language processing. This allows MediGuard to efficiently search for and extract relevant medication details based on user queries.

#### 1.2.1. PDF Processing

The `PyPDFLoader` library is used to load and extract text from each PDF. The PDFs are stored in a directory, and their content is parsed and aggregated into a dictionary, with the filename as the key and the full document text as the value. This allows for structured, efficient access to all available medicine data.

```python
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate

directory = 'medicine_files'
files = [f for f in os.listdir(directory) if f.endswith('.pdf')]

all_medicine_data = {}
```

The text extracted from the PDFs is passed to the language model, which analyzes drug interactions, effects at different doses, effects on different age groups, warnings, and potential adverse effects. This is done by using a `PromptTemplate`, which formats the input and requests JSON-formatted output, making it easier to process the information.

```python
# Function to analyze drug interactions
def analyze_interactions(doc_text):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=(
            "Analyze the following text and extract information about drug interactions, also "
            "Effects of the medicine at different doses.\n"
            "Effects on different age groups (children, adults, elderly).\n"
            "Any warnings or special considerations for people with allergies.\n"
            "Do NOT include any contact information, phone numbers, URLs.\n"
            "including any potential adverse effects or warnings:\n\n{text}\n\n"
            "Provide the results in JSON format."
        )
    )
    prompt = prompt_template.format(text=doc_text)
    response = llm.invoke(prompt)
    return response
```

Since the task of processing multiple PDFs can be time-consuming, a `ThreadPoolExecutor` is used to process the files in parallel, improving efficiency. Each PDF is analyzed and stored in a dictionary, all_medicine_data, which holds the extracted data for future reference.

```python
# Function to process a single PDF file
def process_pdf(file_name):
    file_path = os.path.join(directory, file_name)
    print(f"Starting processing: {file_name}")
    try:
        loader = PyPDFLoader(file_path)
        doc_text = loader.load()
        full_text = "\n".join([doc.page_content for doc in doc_text])

        # Analyze text for interactions
        interaction_info = analyze_interactions(full_text)
        return file_name, interaction_info
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
        return file_name, None

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_pdf, file_name): file_name for file_name in files}
    
    # Use tqdm to show progress
    for future in tqdm(as_completed(futures), total=len(files)):
        file_name, interaction_info = future.result()
        if interaction_info is not None:
            all_medicine_data[file_name] = interaction_info
        else:
            print(f"Failed to analyze interactions for {file_name}")

print("Processing complete.")
```

To optimize the process further, the data is saved in a JSON file after the PDFs are processed. This ensures that the PDFs are only processed once, and in subsequent runs, the JSON file is loaded directly, saving significant time by avoiding reprocessing.

```python
import re
data_file = 'all_medicine_data.json' 

def clean_content(content):
    return re.sub(r'```json\n|```', '', content).strip()

def process_and_save_data(data):
    serializable_data = {}
    
    for key, value in data.items():
        if isinstance(value, AIMessage) and hasattr(value, 'content'):
            try:
                cleaned_content = clean_content(value.content)
                content = json.loads(cleaned_content)
                serializable_data[key.replace('.pdf', '')] = content
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON for key '{key}'. Error: {e}")
                continue
            print(f"Key: {key}, Content: {cleaned_content}")
        else:
            serializable_data[key] = value
            print(f"Key: {key}, Value: {value}")
    
    # Save to JSON file
    with open(data_file, 'w') as file:
        json.dump(serializable_data, file, indent=4)
    print(f"Data saved to {data_file}.")

process_and_save_data(all_medicine_data)
```

In future queries or uses, instead of reprocessing the entire dataset, the saved JSON file is loaded to retrieve the information, significantly reducing data retrieval time and improving overall performance.

```python
# Function to load the medicine data from a file
def load_medicine_data_from_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                medicine_data = json.load(file)
                print(f"Loaded data from {file_path}")
                return medicine_data
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from {file_path}")
            return {}
        except Exception as e:
            print(f"Failed to load data: {e}")
            return {}
    else:
        print(f"File {file_path} not found.")
        return {}

json_file_path = '/nvme/h/hack24gp10/hackathon/all_medicine_data.json'
all_medicine_data = load_medicine_data_from_file(json_file_path)
```

By saving the extracted information in a JSON file, the application ensures that future queries or processing can be handled efficiently by loading the preprocessed data from the file, without needing to reanalyze the entire dataset.

#### 1.2.2. Information Retrieval
The RAG system enhances the medicine information processing by allowing users to query the database for specific medication details. When a user inputs a query, search_medicine_info function scans through the collected documents to find relevant matches. This ensures that MediGuard quickly retrieves the most relevant medication details, improving both accuracy and response time for users.

```python
def search_medicine_info(query, all_medicine_data):
    relevant_docs = []
    query_terms = query.lower().split()

    for file_name, doc_data in all_medicine_data.items():
        if isinstance(doc_data, str):
            try:
                doc_data = json.loads(doc_data)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON for {file_name}")
                continue

        if isinstance(doc_data, dict):
            doc_content = json.dumps(doc_data).lower()
            if any(term in doc_content for term in query_terms):
                relevant_docs.append(doc_data)
        else:
            print(f"Unexpected data format in {file_name}, expected dict but got {type(doc_data)}")

    if not relevant_docs:
        relevant_docs.append({"page_content": "No relevant data found"})

    return relevant_docs
```

## 2. LLM Interaction Checker


### 2.1. Prompt Creation

The LLM Interaction Checker leverages a `PromptTemplate` to guide the LLM in generating accurate drug interaction assessments. The prompt template defines the instructions for the model to analyze the interaction between two drugs, considering patient-specific data such as age, allergies, and health conditions. The severity of interactions is classified into "Major" or "Minor" categories, based on the potential risks to the patient. This ensures that the generated response is precise and aligned with the patient's context.

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """
You are a highly knowledgeable medical expert responsible for analyzing drug-drug interactions with special attention to the patient's condition, age, allergies, and illnesses. 
For each interaction, give a direct, concise explanation based on the severity of the interaction. 
- If there is a serious risk (e.g., overdose, CNS depression, respiratory failure, or death), classify it as 'Major'.
- If the interaction is less serious or lacks strong evidence, classify it as 'Minor'. 
- Do **not** use the phrase "The interaction between ...". Just explain the interaction directly.

Always ensure the severity is aligned with your explanation.

Patient details:
- Age: {age}
- Allergies: {allergies}
- Health condition: {illnesses}

Context: {context}

Please provide a concise explanation of the interaction between '{drug1}' and '{drug2}'.
"""

prompt = PromptTemplate(
    input_variables=["context", "drug1", "drug2"],
    template=template
)
```

### 2.2. Interaction Response Generation

The function ``generate_interaction_response`` searches for relevant drug interaction data and uses the LLM to generate an analysis. It first retrieves documents containing relevant information about the drug pair.  Once data is gathered, the LLM analyzes the interaction in the context of the patient’s health details, providing a concise explanation and classifying the interaction severity as either "Major" or "Minor."

```python
def generate_interaction_response(drug1, drug2, all_medicine_data, llm, patient_age, patient_allergies, patient_illnesses):
    # Search for relevant documents
    relevant_docs = search_medicine_info(f"{drug1} {drug2}", all_medicine_data)

    if not relevant_docs or relevant_docs[0].get("page_content") == "No relevant data found":
        return json.dumps({
            "interaction": [drug1,drug2],
            "severity": "Minor",
            "Explanation": f"No major interactions were found in the dataset, but it's recommended to consult a healthcare professional regarding {drug1} and {drug2}."
        })

    # Create context based on found data
    context = " ".join([doc.get('page_content', '') for doc in relevant_docs])

    try:
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "drug1": drug1,
            "drug2": drug2,
            "age": patient_age,
            "allergies": patient_allergies,
            "illnesses": patient_illnesses
        })

        # Clean up unnecessary symbols and metadata
        response_str = str(response).strip()
        clean_response = response_str.replace("content=", "").replace("\\n", " ").replace("\n", " ").strip()

        if 'additional_kwargs' in clean_response:
            clean_response = clean_response.split('additional_kwargs')[0].strip()        
        if 'response_metadata' in clean_response:
            clean_response = clean_response.split('response_metadata')[0].strip()

        short_explanation = '. '.join(clean_response.split('. ')[:2]).strip()

        # Handle short or vague explanation
        if len(short_explanation) < 30 or "no significant interaction" in short_explanation.lower():
            short_explanation = (f"No significant interactions were found between {drug1} and {drug2}. "
                                 f"However, it's always recommended to consult a healthcare provider to ensure safe use, "
                                 "especially considering individual factors such as age, allergies, or existing conditions.")

        # Determine interaction severity
        if "coma" in short_explanation.lower() or "respiratory distress" in short_explanation.lower() or "overdose" in short_explanation.lower():
            severity = "Major"
        else:
            severity = "Minor"

        if "major" in short_explanation.lower() and severity != "Major":
            severity = "Major"
        short_explanation = short_explanation.replace("**Minor**", "").replace("**Major**", "").strip()
        short_explanation = short_explanation.strip("'").strip("\"")

        # Formulate final JSON response
        return json.dumps({
            "interaction": [drug1,drug2],
            "severity": severity,
            "Explanation": short_explanation
        })

    except Exception as e:
        return json.dumps({
            "interaction": [drug1,drug2],
            "severity": "Minor",
            "Explanation": f"Please consult with a doctor for more details on the interaction between {drug1} and {drug2}."
        })

```

The ``check_drug_drug_interactions`` function identifies potential interactions between a patient's prescribed drugs using parallel processing. It creates unique drug pairs to avoid duplicate checks and efficiently analyzes interactions for each pair. Results, including interaction severity and explanations, are gathered and stored in a dictionary.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_drug_drug_interactions(patient_df, all_medicine_data, llm, patient_id):
    potential_issues = {f"PATIENT {patient_id}": []}
    drug_list = patient_df['Drug_Name'].tolist()
    total_combinations = len(drug_list) * (len(drug_list) - 1)
    
    # Get patient data: age, allergies, and conditions
    patient_age = patient_df.iloc[0]['Age']
    patient_allergies = patient_df.iloc[0].get('Allergies', 'None')
    patient_illnesses = patient_df.iloc[0].get('Illnesses', 'None')
    
    seen_combinations = set()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        print(f"Starting drug interaction checks for patient {patient_id}...")

        for i, drug_name in enumerate(drug_list):
            for j, other_drug in enumerate(drug_list):
                # Skip same drugs
                if i == j or drug_name == other_drug:
                    continue

                # Create sorted pair to avoid duplicate checks like A-B and B-A
                drug_pair = tuple(sorted([drug_name, other_drug]))
                
                # Skip if the pair was already checked
                if drug_pair in seen_combinations:
                    continue

                # Add unique pair
                seen_combinations.add(drug_pair)
                # Pass patient info to generate_interaction_response
                future = executor.submit(generate_interaction_response, drug_name, other_drug, all_medicine_data, llm, patient_age, patient_allergies, patient_illnesses)
                futures.append(future)

        # Process results with a progress bar
        for future in tqdm(as_completed(futures), total=len(seen_combinations), desc="Analyzing drug interactions"):
            model_response = future.result()
            if model_response:
                model_response_dict = json.loads(model_response)
                explanation = model_response_dict.get("Explanation", "No explanation available")

                potential_issues[f"PATIENT {patient_id}"].append({
                    "interaction": model_response_dict["interaction"],
                    "severity": model_response_dict["severity"],
                    "Explanation": explanation
                })

    print(f"Drug interaction analysis for patient {patient_id} completed.")
    return json.dumps(potential_issues, indent=4)
```

Calls `check_drug_drug_interactions` with patient data and returns interaction results as JSON.

```python
def analyze_patient_interactions(patient_df, all_medicine_data, llm, patient_id):
    drug_drug_json = check_drug_drug_interactions(patient_df, all_medicine_data, llm, patient_id)
    return drug_drug_json
```

## 3. Saving Interaction Data

The ``save_all_patient_interactions_to_file`` function collects drug interaction data for each patient and saves it as a JSON file.

```python
def save_all_patient_interactions_to_file(all_patients_df, all_medicine_data, llm, output_file_path):
    all_interactions = {}

    for patient_id in all_patients_df.index.unique():
        patient_df = get_patient(patient_id)
        interactions_json = analyze_patient_interactions_v6(patient_df, all_medicine_data, llm, patient_id)
        all_interactions[f"PATIENT {patient_id}"] = json.loads(interactions_json)[f"PATIENT {patient_id}"]

    # Save all interactions to a JSON file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(all_interactions, file, ensure_ascii=False, indent=4)
        print(f"Interactions data saved successfully to {output_file_path}")
    except Exception as e:
        print(f"Failed to save data: {e}")

output_file_path = '/nvme/h/hack24gp10/hackathon/patient_interactions.json'
save_all_patient_interactions_to_file(all_patients_df, all_medicine_data, llm, output_file_path)
```