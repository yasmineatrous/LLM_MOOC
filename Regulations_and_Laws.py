import streamlit as st  # For creating the web application

import networkx as nx  # For creating and manipulating graphs
from pyvis.network import Network  # For interactive graph visualization
import asyncio  # For asynchronous programming
from termcolor import colored  # For colored console output
import streamlit.components.v1 as components  # For embedding custom HTML in Streamlit
from unified import UnifiedApis  # Custom API wrapper for different AI providers
import PyPDF2  # For reading PDF files
import io  # For handling I/O operations
import os  # For interacting with the operating system
import xml.etree.ElementTree as ET  # For parsing XML
import json  # For working with JSON data
import base64  # For encoding data for download links
from neo4j import GraphDatabase
from typing import List, Dict
import os
from dotenv import load_dotenv
#from langchain.vectorstores import Pinecone as PineconeVectorStore
#from langchain_community.embeddings import SentenceTransformerEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import PyPDFLoader  
from dotenv import load_dotenv
#from langchain_openai import OpenAIEmbeddings
import tempfile
#from pinecone import Pinecone
import warnings
import pickle 
import streamlit_authenticator as stauth 
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Regulation and Laws Upload", page_icon="🔗", layout="wide")

# Load environment variables
load_dotenv()

# Set up environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI") 
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize embeddings and Pinecone
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index
#index_name = "hack"
#index = pc.Index(index_name)
# Print index statistics
#index.describe_index_stats()
import streamlit as st
import pickle
import streamlit_authenticator as stauth 

# Load user data
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

st.markdown("""
<style>
    
    /* Make sure the sidebar logout button is at the bottom */
    [data-testid="stSidebar"] .css-ng1t4o {
        display: flex;
        flex-direction: column;
        height: 100%; /* Ensure it spans the full height */
    }
    [data-testid="stSidebar"] .css-ng1t4o > div:last-child {
        margin-top: auto; /* Push the last child (logout) to the bottom */
    }
</style>
""", unsafe_allow_html=True)
# load hashed passwords
file_path = "./hashed_pw.pkl"
with open(file_path, "rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "pde", "abcde", cookie_expiry_days=30)

st.markdown("""
<style>
    /* Hide Streamlit default elements before login */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Optional: Hide other Streamlit UI elements */
    header, footer {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)
name, authentication_status, username = authenticator.login("Login", "main")
#authenticator.logout("Logout", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")
elif authentication_status:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: block !important;
        }
        header, footer {
            display: block !important;
        }
    </style>
    """, unsafe_allow_html=True)
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")



    class Neo4jManager:
        def __init__(self, uri: str, username: str, password: str):
            """
            Initialize a connection to the Neo4j database.
            
            :param uri: Neo4j database connection URI
            :param username: Neo4j username
            :param password: Neo4j password
            """
            self._driver = GraphDatabase.driver(uri, auth=(username, password))

        def close(self):
            """Close the database connection."""
            self._driver.close()

        def clear_graph(self):
            """Clear all nodes and relationships in the graph."""
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")

        def create_knowledge_graph(self, entities: List[str], relations: List[Dict], document_id: str):
            """
            Create a knowledge graph in Neo4j based on extracted entities and relations.
            
            :param entities: List of entity names
            :param relations: List of relation dictionaries
            :param document_id: Unique identifier for the source document
            """
            with self._driver.session() as session:
                # Create nodes for each entity with document source
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET e.documents = [$document_id]
                        ON MATCH SET e.documents = CASE 
                            WHEN NOT $document_id IN e.documents 
                            THEN e.documents + $document_id 
                            ELSE e.documents 
                        END
                        """,
                        name=entity,
                        document_id=document_id
                    )

                # Create relationships between entities
                for relation in relations:
                    session.run(
                        """
                        MATCH (source:Entity {name: $source_name})
                        MATCH (target:Entity {name: $target_name})
                        MERGE (source)-[r:RELATES {type: $relation_type, documents: [$document_id]}]->(target)
                        ON CREATE SET r.first_seen = $document_id
                        ON MATCH SET r.documents = CASE 
                            WHEN NOT $document_id IN r.documents 
                            THEN r.documents + $document_id 
                            ELSE r.documents 
                        END
                        """,
                        source_name=relation['source'],
                        target_name=relation['target'],
                        relation_type=relation['type'],
                        document_id=document_id
                    )

        def get_entities_and_relations(self):
            """
            Retrieve all entities and relations from the Neo4j database.
            
            :return: Tuple of (entities, relations)
            """
            with self._driver.session() as session:
                # Fetch all entities
                entities_result = session.run("MATCH (e:Entity) RETURN e.name as name, e.documents as documents")
                entities = [
                    {
                        'name': record['name'], 
                        'documents': record['documents']
                    } for record in entities_result
                ]

                # Fetch all relations
                relations_result = session.run(
                    "MATCH (source)-[r:RELATES]->(target) " 
                    "RETURN source.name as source, target.name as target, r.type as type, r.documents as documents"
                )
                relations = [
                    {
                        'source': record['source'],
                        'target': record['target'],
                        'type': record['type'],
                        'documents': record['documents']
                    } for record in relations_result
                ]

                return entities, relations
    # Print a colored message to indicate the start of the application
    print(colored("Initializing Knowledge Graph Visualizer with File Upload...", "cyan"))
    async def get_entities_and_relations(api, text):
        system_message = """
        You are an expert in natural language processing and knowledge extraction, specializing in regulatory and compliance texts.

       Given a document or passage, extract the main rules, regulations, and their associated entities . 
       Identify the relationships between these entities, ensuring that all rules and their connections are clearly linked.

        Given a text, identify the main entities and their relationships.
        Return your response in the following XML format:

        <output>
        <entities>
        <entity>Entity1</entity>
        <entity>Entity2</entity>
        ...
        </entities>
        <relations>
        <relation>
        <source>SourceEntity</source>
        <target>TargetEntity</target>
        <type>RelationType</type>
        </relation>
        ...
        </relations>
        </output>

        Ensure that the XML is well-formed and does not contain any syntax errors. You must absolutely at all times return your response in the format presented regardless of how large the document is. If the document is long and overwhelming, then still do your best in returning as accurate information as possible without making a mistake.
        """
        await api.set_system_message_async(system_message)
        try:
            response = await api.chat_async(f"Here is the text which you will be analyzing: {text}\nExtract entities and relations from this text in exactly the format presented")
            
            return response
        except Exception as e:
            return f"<error>{str(e)}</error>"


    async def get_json_entities(api, text):
        print(colored("Extracting Entities in JSON format...", "blue"))
        system_message = """
        You are an expert in entity extraction and classification from PDF documents. Your task is to extract structured entities focusing on rules a large language model should not violate based on the document content with a hierarchical tree structure.

        The JSON structure must list multiple rules, each with its associated violations nested under it.
        make sure to extract the rules and violations in the form of "DO NOT" phrases that a LLM should comply with.
        For each rule, provide the following structured JSON format:
        {
            "rule": "DO not ",
            "violations": [
                {
                    "violation": "Do not violation",
                    "section": "Document Section",
                    "relevant_data": "Supporting data or references from the text",
                    "risk": "critical" or "high" or "medium" or "low"
                },
                ...
            ]
        }

        Return your response always in the following JSON format:
        <output>
        {
            "rules": [
                {
                    "rule": "Do not Rule 1",
                    "violations": [
                        {
                            "violation": "Do not violation",
                            "section": "Document Section",
                            "relevant_data": "Supporting data or references from the text",
                            "risk": "critical" or "high" or "medium" or "low"
                        }
                    ]
                },
                {
                    "rule": "Do not Rule 2",
                    "violations": [
                        {
                            "violation": "do not violation",
                            "section": "Document Section",
                            "relevant_data": "Supporting data or references from the text",
                            "risk": "critical" or "high" or "medium" or "low"
                        }
                    ]
                }
            ]
        }
        </output>

        Ensure the JSON is well-formed, and each rule has its violations correctly nested.
        """


        await api.set_system_message_async(system_message)
        response = await api.chat_async(f"Extract and classify entities from this text in the specified JSON format: {text}")
        json_start = response.find('<output>') + len("<output>")
        json_end = response.rfind("</output>")
        json_content = response[json_start:json_end].strip()

        try:
            # Load existing JSON data if the file exists
            try:
                with open("rules_and_violations.json", "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {"rules": []}

            # Parse the new JSON content
            new_data = json.loads(json_content)
            
            # Append new data to existing rules
            existing_data["rules"].extend(new_data["rules"])

            # Save the updated content back to the file
            with open("rules_and_violations.json", "w", encoding="utf-8") as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
            
            print(colored("JSON content saved to rules_and_violations.json", "green"))
            return json.dumps(existing_data)  # Convert the dict back to a JSON-formatted string for returning
        except Exception as e:
            print(colored(f"Error saving JSON: {e}", "red"))

        return json_content


    def parse_output(output):
        try:
            print(output)
            xml_start = output.find("<output>")
            xml_end = output.find("</output>") + 9  # Length of "</output>"
            if xml_start == -1 or xml_end == -1:
                raise ValueError("XML tags not found in the output")
            xml_content = output[xml_start:xml_end]
            root = ET.fromstring(xml_content)

            entities = [entity.text for entity in root.find('entities')]
            relations = [
                {
                    'source': relation.find('source').text,
                    'target': relation.find('target').text,
                    'type': relation.find('type').text
                }
                for relation in root.find('relations')
            ]

            return entities, relations

        except ValueError as e:
            print(f"Error parsing output: {e}")
            return [], []  

        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return [], []  
        
        except ValueError as e:
            # Handle value errors (e.g., missing XML tags)
            st.error(f"Error: {e}")
            st.error("Raw output:")
            st.code(output)
            return [], []
        
    import networkx as nx
    from pyvis.network import Network

    def create_graph(entities, relations):
        """
        Creates a networkx Graph object from a list of entities and relations.

        Args:
            entities (list): A list of entities (nodes) in the graph.
            relations (list): A list of dictionaries, where each dictionary represents a relation 
                            and has keys 'source', 'target', and 'type'.

        Returns:
            networkx.Graph: The created graph object.
        """

        G = nx.Graph()

        for entity in entities:
            G.add_node(entity)

        for relation in relations:
            G.add_edge(relation['source'], relation['target'], type=relation["type"])

        return G

    def visualize_graph(G):
        """
        Visualizes a networkx Graph object using pyvis.

        Args:
            G (networkx.Graph): The graph to be visualized.

        Returns:
            str: HTML content of the generated graph visualization.
        """

        net = Network(notebook=True, width="100%", height="600px")

        for node in G.nodes():
            net.add_node(node)

        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], title=edge[2]['type'])

        net.save_graph("graph.html")

        with open("graph.html", "r", encoding="utf-8") as f:
            graph_html = f.read()

        return graph_html

    def read_pdf(file):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        print(text)
        return text

    # Function to create a download link for content
    def get_download_link(content, filename, text):
        b64 = base64.b64encode(content.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    async def main():
        # Set up the Streamlit page

        
        #st.set_page_config(layout="wide")
        st.title("🔗 Regulations and Laws Upload")
        st.markdown(
        "Upload multiple documents to extract rules and policies, and generate a dynamic knowledge graph for better explainibility and visualization ."
    )
        # Initialize session state for tracking uploads
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []

        # Check for API keys in environment variables
        #anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("OPENROUTER_API_KEY")

        # Allow user to select the AI model
        model = st.selectbox("Select model:", ["GPT-4", "Gemini 1.5 Pro"])

        # Prompt for API key if not found in environment variables
        #if model == "Claude-3-5-Sonnet" and not anthropic_api_key:
            #anthropic_api_key = st.text_input("Enter your Anthropic API key:", type="password")
        if model == "GPT-4" and not openai_api_key:
            openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        elif model == "Gemini 1.5 Pro" and not google_api_key:
            google_api_key = st.text_input("Enter your Google API key:", type="password")

        # Set up the API clients
        #api_key = openai_api_key
        #provider = "openai"
        api_key = openai_api_key if model == "GPT-4" else  google_api_key
        provider = "openai" if model == "GPT-4" else  "openrouter"

        print(colored(f"Initializing UnifiedApis with model: {model}", "yellow"))

        api_kg = UnifiedApis(name="knowledge_graph", api_key=api_key, provider=provider, use_async=True, max_history_words=100000)
        api_summary = UnifiedApis(name="summary", api_key=api_key, provider=provider, use_async=True, max_history_words=100000)
        api_json = UnifiedApis(name="json_entities", api_key=api_key, provider=provider, use_async=True, max_history_words=100000)


        # Initialize Neo4j Manager
        try:
            neo4j_manager = Neo4jManager(neo4j_uri, neo4j_username, neo4j_password)
            print("Neo4j connected successfully!")
        except Exception as e:
            st.error(f"Neo4j Connection Error: {e}")
            neo4j_manager = None

        # Allow user to upload multiple files
        #input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])


        #if input_method == "Paste Text":
            #text = st.text_area("Enter your text here:")

        #else:  # Upload File
            #uploaded_files = st.file_uploader("Choose a file", type=["txt", "pdf"], accept_multiple_files=True)
        uploaded_files = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)   
            # Debugging
        #if not uploaded_files:
                #st.warning("No files uploaded. Please upload a file to proceed.")
        #else:
                #st.write(f"Files uploaded: {[file.name for file in uploaded_files]}")


        # Options for graph management
        if st.sidebar.button("Clear Graph"):
            if neo4j_manager:
                try:
                    neo4j_manager.clear_graph()
                    st.session_state.uploaded_documents.clear()
                    st.success("Graph cleared successfully!")
                    rules_file_path = "rules_and_violations.json"
                    if os.path.exists(rules_file_path):
                        os.remove(rules_file_path)
                    else:
                        st.warning("No rules file found to delete.")
                except Exception as e:
                    st.error(f"Error clearing graph: {e}")    

        if st.sidebar.button("Generate Graph"):
            col1, col2 = st.columns(2)
            with col1:
            
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        # Skip already processed documents
                        if uploaded_file.name in st.session_state.uploaded_documents:
                            continue

                        #if uploaded_file.type == "text/plain":
                            #text = uploaded_file.getvalue().decode("utf-8")
                        if uploaded_file.type == "application/pdf":
                            with st.spinner(f"Loading PDF: {uploaded_file.name}..."):
                                text = read_pdf(uploaded_file)
                        
                        # Initialize APIs
                        api_kg = UnifiedApis(name="knowledge_graph", api_key=api_key, provider=provider, use_async=True, max_history_words=100000)
                        
                        # Extract entities and relations
                        with st.spinner(f"Extracting entities from {uploaded_file.name}..."):
                            output = await get_entities_and_relations(api_kg, text)
                            entities, relations = parse_output(output)

                        # Store in Neo4j with document tracking
                        if neo4j_manager and entities and relations:
                            try:
                                neo4j_manager.create_knowledge_graph(entities, relations, uploaded_file.name)
                                st.session_state.uploaded_documents.append(uploaded_file.name)
                            except Exception as e:
                                st.error(f"Error storing graph for {uploaded_file.name}: {e}")

                        with st.spinner("Extracting entities in JSON format..."):
                            json_entities = await get_json_entities(api_json, text)
                            st.subheader("Entities (JSON):")

                            try:
                                parsed_json = json.loads(json_entities)
                                st.json(parsed_json)

                                # Provide a download link for the JSON data
                                st.markdown(get_download_link(json.dumps(parsed_json, indent=2), "entities.json","Download JSON"), unsafe_allow_html=True)
                            except json.JSONDecodeError:
                                st.error("Failed to parse JSON. Raw output:")
                                st.code(json_entities)

            with col2:


            # Visualization of Accumulated Graph
                    if neo4j_manager:
                        try:
                            # Retrieve stored entities and relations
                            stored_entities, stored_relations = neo4j_manager.get_entities_and_relations()

                            # Extract just the names for graph creation
                            entity_names = [entity['name'] for entity in stored_entities]

                            # Create graph visualization
                            G = create_graph(entity_names, stored_relations)
                            graph_html = visualize_graph(G)

                            st.subheader("Cumulative Knowledge Graph:")
                            components.html(graph_html, height=600)
                            st.markdown(get_download_link(graph_html, "knowledge_graph.html", "Download Knowledge Graph"), unsafe_allow_html=True)

                            # Display document sources for entities
                            #st.subheader("Entity Sources:")
                            #for entity in stored_entities:
                                #st.write(f"{entity['name']}: Documents {entity.get('documents', [])}")

                            #st.subheader("Relation Details:")
                            #for relation in stored_relations:
                                #st.write(f"{relation['source']} -> ({relation['type']}) -> {relation['target']} (Docs: {relation.get('documents', [])})")


                        except Exception as e:
                            st.error(f"Error visualizing graph: {e}")
                    else:
                        st.error("Neo4j connection not established.")


                            

 


    if __name__ == "__main__":
        print(colored("Starting Cumulative Knowledge Graph", "magenta"))
        asyncio.run(main())