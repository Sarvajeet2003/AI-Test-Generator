import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import csv
from typing import Dict, List, Union
import os
import io

class DataDictionaryTestCaseGenerator:
    """
    AI Agent that reads data dictionary from file and generates test cases using Gemini-1.5-flash
    """
    
    def __init__(self, api_key: str):
        """Initialize the AI agent with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def load_data_dictionary_from_uploaded_file(self, uploaded_file) -> Dict:
        """Load data dictionary from uploaded file"""
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                return self._load_from_csv_file(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        except Exception as e:
            raise Exception(f"Error loading data dictionary: {str(e)}")
    
    def _load_from_csv_file(self, uploaded_file) -> Dict:
        """Load data dictionary from uploaded CSV file"""
        df = pd.read_csv(uploaded_file)
        
        # Expected CSV columns: field_name, data_type, description, allowed_values, required, example
        required_columns = ['field_name', 'data_type']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        data_dict = {}
        for _, row in df.iterrows():
            field_name = row['field_name']
            data_dict[field_name] = {
                'data_type': row.get('data_type', ''),
                'description': row.get('description', ''),
                'allowed_values': row.get('allowed_values', ''),
                'required': row.get('required', False),
                'example': row.get('example', '')
            }
        
        return data_dict
    
    def generate_test_cases(self, data_dictionary: Dict) -> List[Dict]:
        """Generate test cases based on data dictionary using AI"""
        
        prompt = f"""
        Analyze this data dictionary and generate comprehensive test cases for data validation.
        
        Data Dictionary:
        {json.dumps(data_dictionary, indent=2)}
        
        For each field, generate test cases that cover:
        1. Data type validation
        2. Required field validation (if applicable)
        3. Format/pattern validation
        4. Range/constraint validation
        5. Allowed values validation
        
        Return the result as a JSON array where each object has this structure:
        {{
            "field_name": "field_name",
            "test_case_id": "TC_001",
            "test_case_name": "descriptive test name",
            "test_description": "what this test validates",
            "test_type": "data_type|required|format|range|allowed_values",
            "validation_rule": "specific validation rule to check",
            "expected_result": "what should happen if test passes"
        }}
        
        Return only the JSON array, no explanations or markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean the response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Parse JSON response
            test_cases = json.loads(response_text)
            return test_cases
            
        except Exception as e:
            st.error(f"AI generation failed: {e}")
            return self.generate_fallback_test_cases(data_dictionary)
    
    def generate_fallback_test_cases(self, data_dictionary: Dict) -> List[Dict]:
        """Generate basic test cases when AI fails"""
        test_cases = []
        for i, (field_name, field_info) in enumerate(data_dictionary.items(), 1):
            test_case = {
                "field_name": field_name,
                "test_case_id": f"TC_{i:03d}",
                "test_case_name": f"Validate {field_name} data type",
                "test_description": f"Validate that {field_name} has correct data type",
                "test_type": "data_type",
                "validation_rule": f"Check if {field_name} is of type {field_info.get('data_type', 'unknown')}",
                "expected_result": "Field should have correct data type"
            }
            test_cases.append(test_case)
        return test_cases

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.set_page_config(
        page_title="Test Case Generator Dashboard",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Data Dictionary Test Case Generator")
    st.markdown("Upload your data dictionary file and generate comprehensive test cases using AI")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Enter Gemini API Key:", type="password", value="AIzaSyCEd6f7y1XTUG4P42tEwSdT1_Nf-h76sRs")
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Enter your Gemini API key
        2. Upload a CSV file with your data dictionary
        3. CSV should have columns: `field_name`, `data_type`
        4. Optional columns: `description`, `allowed_values`, `required`, `example`
        5. Click 'Generate Test Cases' to create tests
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Data Dictionary")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your data dictionary CSV file"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Display file preview
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.subheader("📊 File Preview")
                st.dataframe(df_preview.head(), use_container_width=True)
                st.info(f"Total fields: {len(df_preview)}")
                
                # Reset file pointer for processing
                uploaded_file.seek(0)
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.header("🚀 Generate Test Cases")
        
        if uploaded_file is not None and api_key:
            if st.button("Generate Test Cases", type="primary", use_container_width=True):
                try:
                    with st.spinner("🤖 AI is analyzing your data dictionary..."):
                        # Initialize generator
                        generator = DataDictionaryTestCaseGenerator(api_key)
                        
                        # Load data dictionary
                        uploaded_file.seek(0)
                        data_dictionary = generator.load_data_dictionary_from_uploaded_file(uploaded_file)
                        
                        st.success(f"✅ Loaded {len(data_dictionary)} fields from data dictionary")
                        
                        # Generate test cases
                        test_cases = generator.generate_test_cases(data_dictionary)
                        
                        # Store in session state
                        st.session_state.test_cases = test_cases
                        st.session_state.data_dictionary = data_dictionary
                        
                        st.success(f"✅ Generated {len(test_cases)} test cases")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            if not api_key:
                st.warning("⚠️ Please enter your Gemini API key")
            if not uploaded_file:
                st.warning("⚠️ Please upload a data dictionary file")
    
    # Display results if available
    if 'test_cases' in st.session_state:
        st.header("📊 Generated Test Cases Dashboard")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Test Cases", len(st.session_state.test_cases))
        with col2:
            test_types = [tc['test_type'] for tc in st.session_state.test_cases]
            st.metric("Unique Test Types", len(set(test_types)))
        with col3:
            fields = [tc['field_name'] for tc in st.session_state.test_cases]
            st.metric("Fields Covered", len(set(fields)))
        with col4:
            st.metric("Data Dictionary Fields", len(st.session_state.data_dictionary))
        
        # Test cases table
        st.subheader("📋 Test Cases Details")
        df_test_cases = pd.DataFrame(st.session_state.test_cases)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_fields = st.multiselect(
                "Filter by Field:",
                options=df_test_cases['field_name'].unique(),
                default=df_test_cases['field_name'].unique()
            )
        with col2:
            selected_types = st.multiselect(
                "Filter by Test Type:",
                options=df_test_cases['test_type'].unique(),
                default=df_test_cases['test_type'].unique()
            )
        
        # Apply filters
        filtered_df = df_test_cases[
            (df_test_cases['field_name'].isin(selected_fields)) &
            (df_test_cases['test_type'].isin(selected_types))
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download section
        st.subheader("💾 Download Test Cases")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = convert_df_to_csv(df_test_cases)
            st.download_button(
                label="📥 Download All Test Cases (CSV)",
                data=csv_data,
                file_name="generated_test_cases.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            filtered_csv = convert_df_to_csv(filtered_df)
            st.download_button(
                label="📥 Download Filtered Test Cases (CSV)",
                data=filtered_csv,
                file_name="filtered_test_cases.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Test case distribution chart
        st.subheader("📈 Test Case Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            type_counts = df_test_cases['test_type'].value_counts()
            st.bar_chart(type_counts)
            st.caption("Test Cases by Type")
        
        with col2:
            field_counts = df_test_cases['field_name'].value_counts()
            st.bar_chart(field_counts)
            st.caption("Test Cases by Field")

if __name__ == "__main__":
    main()