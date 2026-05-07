# 🚀 Running the Autonomous Ingestion Quality Agent

Clone the repository:

```bash
git clone <your-repo-url>
cd ingestion_quality_agent
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama from:

https://ollama.com/download

Pull the LLM model:

```bash
ollama pull llama3
```

Start the Ollama server:

```bash
ollama serve
```

Keep this terminal running.

Open a new terminal inside the project directory and start the Streamlit application:

```bash
streamlit run app.py
```

Open the Streamlit URL shown in the terminal and upload any raw CSV file.

The autonomous pipeline will:
- profile the dataset
- infer semantic structure
- generate validation rules
- detect anomalies
- self-heal corrupted data
- quarantine unsafe rows
- return clean downloadable datasets

The system generates:
- clean processed CSV
- quarantine CSV
- validation reasoning trace
- autonomous healing logs
- generated expectation suite
