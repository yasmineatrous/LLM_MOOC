import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import warnings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Set up environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize embeddings and Pinecone
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY)
# Define Pinecone index
st.set_page_config(page_title="Safeguarding Your Rag", page_icon="🔍", layout="centered")
#st.title("🔍 Query Safeguard LLM")

# Function to retrieve relevant context from Pinecone
def retrieve_from_pinecone(user_query, index_name):
    pinecone = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    context = pinecone.similarity_search(user_query)
    return context

import json
with open('../rules_and_violations.json', 'r') as file:
    data = json.load(file)
def check_category(user_query):
    data_text = json.dumps(data, indent=2)
    messages = [
        ("system", f"""
    This is a set of do not rules you must follow and comply with before answering a query: {data_text}
    Your task is to evaluate if your response to the following user query would violate or go against any of those rules.
    User Query: "{user_query}"
    
    You must consider your **own response** to this query, not the query itself, and check if it would violate any of the rules.
    If your response to this query would violate any of the rules, tell me which specific rule is violated and explain why. 
    Check the user query and identify which rule(s) may be violated in your response.
    if it does tell me which rule is violated and why.
    """),
        ("human", ""),
    ]


    decision = llm.invoke(messages) 
    #st.write( decision.content  ) # Get the first result from the generator
    return decision.content 


def should_answer(user_query, context):
    

    messages = [
    ("system", f"""
        You are an AI model designed to assist with queries while adhering to specific guidelines.
        
        This text outlines whether or not a rule has been violated: {context}
        
        Based on the text, decide whether the query should be answered or not. 
        If a rule has been violated, reply with "NO". If no rule was violated and it looks safe to answer, reply with "YES".

        Your response must be strictly "YES" or "NO" without any additional text or explanation.
        User Query: {user_query}
    """),
    ("human", f"""
        Respond strictly with either "YES" or "NO". Do not include any explanation or additional information.
    """),
]

    decision = llm.invoke(messages)  # Get the first result from the generator
    return decision.content.upper()   # Now, you can safely apply .upper() to the result

def extract_rule_and_risk_category(user_query, rules):
    

    messages = [
    ("system", f"""
        You are an AI that determines what exact rule was most likely violated from the rules list.
        The rule: {rules}
        The user query: {user_query}
        
        What rule is the rule that was violated by the query? Answer in JSON format:

        {{
            "rule": "DO not ",
            "violations": [
                {{
                    "violation": "Do not violation",
                    "section": "Document Section",
                    "relevant_data": "Supporting data or references from the text",
                    "risk": "critical" or "high" or "medium" or "low"
                }}
            ]
        }}
    """),
    ("human", f"""
    """),
]


    decision = llm.invoke(messages)  # Get the first result from the generator
    return decision.content

# Function to save violation log to a file
def save_violations_to_file():
    with open("../violations.json", "w") as file:
        json.dump(st.session_state.violation_log, file, indent=4)

# Define the function for getting responses
def get_response(user_query):
    context2 = check_category(user_query)
    
    st.session_state.context_log = [context2]
    output = should_answer(user_query, context2)
    

    if output.upper() == "YES":
        index_name1 = "rag"
        context1 = retrieve_from_pinecone(user_query, index_name1)[:5]
        template = """
    You are an RAG AI assistant. Answer the question below using this context retrieved from a vector database: {context1}.
    Query: {user_question}
    If the context does not contain relevant details about the question, answer it directly using your own knowledge without providing any explanation or justification.
"""


        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        st.success("✅ Safe: This content is appropriate for the RAG to answer.")
        # Returning response as a generator
        return chain.stream({
            "context1": context1,
            "user_question": user_query
        })

    elif output.upper() == "NO":
        data_text = json.dumps(data, indent=2)

        Violation = extract_rule_and_risk_category(user_query, data_text)
        st.session_state.violation_log.append(f"Violated Rule: {Violation}")
        save_violations_to_file()  # Save violations to file whenever updated
        template = """
            From the Rule list, give the exact Rule this Query violated.
            Rules: {context2}
            User question: {user_question}
            Respond to the user question with: Sorry, I can Not answer this question because this query violates the following Rule: [write the rule here].
            example: Sorry, I cannot answer this question because this query violates the Rule: Complete Exclusion of Cat-Related Content.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        st.error("❌ Unsafe: This content may not be appropriate for the RAG to answer.")
        # Returning response as a generator
        return chain.stream({
            "context2": context2,
            "user_question": user_query
        })
    else:
        st.error("Unexpected decision from LLM. Please review rules or query.")
st.markdown("""
<style>
    .stApp {
        background-color: #f4f4f8;
        font-family: 'Inter', sans-serif;

    }

    .chat-header {
        text-align: center;
        margin-bottom: 20px;
        color: #333;
        
    }
    .chat-message {
        margin-bottom: 15px;
        padding: 12px;
        border-radius: 8px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .human-message {
        background-color: #e6f2ff;
        border-left: 4px solid #3182ce;
    }
    .ai-message {
        background-color: #f0f4f8;
        border-left: 4px solid #48bb78;
    }
    .violation-log {
        background-color: #FFE5B4; 
        border-left: 4px solid #FF6B6B; 
        padding: 10px; 
        margin-bottom: 10px; 
        border-radius: 4px;
    }
    /* Added important styles to ensure they persist */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #e0e0e0 !important;
    }
    .stForm > div > div {
        display: flex;
        flex-direction: column;
    }
    .stButton > div > button {
        display: block !important;
    }

</style>
""", unsafe_allow_html=True)

#st.markdown("<h1 class='chat-header'>🛡️ Query Safeguard LLM</h1>", unsafe_allow_html=True)

# Initialize chat history and context
if "violation_log" not in st.session_state:
    st.session_state.violation_log = []  # Store the violated rules
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi, I'm a safeguarded LLM. How can I help you?")]

def display_violation_log():
    if st.session_state.violation_log:
        st.markdown("### 🚨 Violation Log")
        for i, violation in enumerate(st.session_state.violation_log, 1):
            st.markdown(f"""
            <div class="violation-log">
                <strong>Violation #{i}:</strong> {violation}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No rule violations detected yet.")



   
#st.markdown("<h1 class='chat-header'>🛡️ Query Safeguard LLM</h1>", unsafe_allow_html=True)

# Violation log toggle
show_violations = st.toggle("Show Violation Log")
if show_violations:
    display_violation_log()

# Chat history display
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI" , avatar="../bot.png"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        st.markdown(f"""
        <div class='chat-message human-message'>
        {message.content}
        </div>
        """, unsafe_allow_html=True)

# User input
user_query = st.chat_input("Type your message here...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

        
        # Display user message
    st.markdown(f"""<div class='chat-message human-message'>{user_query}</div>""", unsafe_allow_html=True)

    with st.chat_message("AI",avatar="../bot.png"):
        response = st.write_stream(get_response(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))


