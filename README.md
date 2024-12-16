# SecureRAG: Regulatory Compliance-Aware RAG System

## Overview

SecureRAG is an advanced Retrieval-Augmented Generation (RAG) system designed to ensure regulatory compliance and data protection across various industries. By integrating Large Language Models (LLMs), Knowledge Graphs, and sophisticated compliance checking, SecureRAG provides a secure and transparent approach to information retrieval.

## Key Features

- 🔒 **Regulatory Compliance**: Comprehensive compliance checks against uploaded regulations
- 🧠 **Knowledge Graph Integration**: Transforms regulatory documents into a structured, navigable format
- 🚨 **Real-time Violation Detection**: Immediate blocking and logging of non-compliant queries
- 👀 **Admin Oversight**: Detailed violation dashboard for continuous improvement

## Prerequisites

- Python 3.9+
- Requirements listed in `requirements.txt`

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/SecureRAG.git
   cd SecureRAG
2.Create a virtual environment
```bash
  python -m venv venv
  source venv/bin/activate
3.Install requirements
```bash
   pip install -r requirements.txt
4.Set up environment variables
Add necessary configuration variables to the .env file

## Running the Application
Admin Interface
```bash
   streamlit run Regulations_and_Laws.py
## Sample Credentials to run
Username: pparker
Password: abc

## User RAG Interface
streamlit run rag/safe_guard_rag.py

