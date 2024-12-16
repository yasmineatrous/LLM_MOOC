# SecureRAG: A Secure RAG System for Compliance and Security 🌐🔐

**SecureRAG** is a cutting-edge system designed to ensure compliance with global regulations while safeguarding against breaches. By leveraging a combination of a Large Language Model (LLM) and Knowledge Graphs, SecureRAG provides a secure and transparent framework for organizations to manage and evaluate their compliance with various regulations such as GDPR, HIPAA, and more.

## Getting Started 🚀

To get SecureRAG up and running, follow these steps:

### 1. Install the Required Dependencies 📦
First, ensure you are using Python version **>= 3.9**. Then, install the necessary libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Configure the Environment Variables ⚙️
Create a `.env` file and define the necessary environment variables. This file will contain the configuration for your database, API keys, and other required settings.

### 3. Access the Admin Interface 🔑
To manage SecureRAG and monitor compliance, you can access the **Admin Interface** using **Streamlit**:

```bash
streamlit run Regulations_and_Laws.py
```

Login credentials for the admin interface (for testing purposes):
- **Username:** pparker
- **Password:** abc

### 4. Access the User Interface 🖥️
Then To use the SecureRAG system as an end-user, run the following command:

```bash
streamlit run rag/safe_guard_rag.py
```

This will allow users to interact with the system, submit queries, and receive compliance-assessed responses.

---

## Features ✨
- **Automated Compliance Checking**: Ensures that queries comply with global regulations.
- **Knowledge Graph**: Visualizes relationships between different regulatory rules.
- **Human Oversight**: Admin dashboard for continuous monitoring and rule refinement.
- **Real-Time Feedback**: Provides immediate feedback to users about compliance violations.

---

## License 📄

SecureRAG is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---
