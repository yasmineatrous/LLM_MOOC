import streamlit as st
import json
import pandas as pd
import re  # To help with text processing

# Set the page configuration
#st.set_page_config(page_title="Monitoring Dashboard",page_icon="📊",layout="centered", layout="wide")
st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="📊",
    layout="wide",

)

# Load the JSON data
try:
    with open("violations.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    st.error("violations.json file not found. Please ensure it exists in the app directory.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding the JSON file. Please check the file format.")
    st.stop()

# Function to clean and extract data from the text-based violations
def extract_violation_data(entry):
    # Regular expressions to extract key-value pairs
    rule_match = re.search(r'"rule":\s?"(.*?)"', entry)
    violation_match = re.search(r'"violation":\s?"(.*?)"', entry)
    section_match = re.search(r'"section":\s?"(.*?)"', entry)
    relevant_data_match = re.search(r'"relevant_data":\s?"(.*?)"', entry)
    risk_match = re.search(r'"risk":\s?"(.*?)"', entry)

    # If a violation is found, construct a dictionary
    if rule_match and violation_match and section_match and relevant_data_match and risk_match:
        return {
            "Rule": rule_match.group(1),
            "Violation": violation_match.group(1),
            "Section": section_match.group(1),
            "Relevant Data": relevant_data_match.group(1),
            "Risk": risk_match.group(1)
        }
    return None  # If any field is missing, return None

# List to hold extracted violation data
violations = []

# Loop over the entries and clean/extract data
for item in data:
    try:
        # Extract violation data from each string entry
        violation_data = extract_violation_data(item)
        if violation_data:
            violations.append(violation_data)
    except Exception as e:
        st.error(f"Error processing an entry: {e}")
        continue

# Convert to DataFrame
df = pd.DataFrame(violations)

# Check if 'Risk' column exists
if 'Risk' not in df.columns:
    st.error("'Risk' column is missing from the data. Please check the input file.")
    st.stop()



# Title and description
st.title("📊 Monitoring Dashboard")
st.markdown("""
Monitor and analyze rule violations in real time.Explore the rules, sections, risk levels, and more to better understand the detected violations.
""")

# Show summary metrics with tooltips for context
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Summary of Violations")

# Split summary into two columns
col1, col2 = st.columns(2)
with col1:
    total_violations = df.shape[0]
    st.metric(label="Total Violations", value=total_violations, help="Total number of rule violations.")
with col2:
    high_risk_count = df[df["Risk"] == "high"].shape[0]
    st.metric(label="High Risk Violations", value=high_risk_count, help="Number of high-risk violations detected.")

# Actionable Insights or Suggestions
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Alerts and State")
if high_risk_count > 3:
    st.warning("⚠️ **Too many High-risk violations detected!** Immediate action is required to address these issues.")
else:
    st.success("✅ No high-risk violations found. The Users seems to be in good compliance.")

# Filter by Risk Level with an option for searching/filtering
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Filter Violations by Risk Level")
risk_filter = st.multiselect(
    "Select Risk Levels",
    options=df["Risk"].unique(),
    default=df["Risk"].unique(),
    help="Select one or more risk levels to filter the violations."
)
filtered_df = df[df["Risk"].isin(risk_filter)]

st.dataframe(filtered_df)

# Interactive Bar Chart for Violations by Risk Level
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Violations by Risk Level")

# Group by Risk Level and count the occurrences
risk_counts = df['Risk'].value_counts().reset_index()
risk_counts.columns = ['Risk', 'Count']

# Create a simple bar chart with Streamlit's built-in plotting tools
st.bar_chart(risk_counts.set_index('Risk'))


