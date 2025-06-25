import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import csv
from typing import Dict, List, Union
import os
import io

class DataDictionaryTestCaseGenerator:
    """AI Agent that reads data dictionary from file and generates test cases using Gemini-1.5-flash"""

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
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
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

    def execute_test_cases(self, test_cases: List[Dict], dataset: pd.DataFrame) -> pd.DataFrame:
        """Execute test cases on the dataset using AI"""
        results = []
        for test_case in test_cases:
            field_name = test_case['field_name']
            if field_name not in dataset.columns:
                results.append({
                    **test_case,
                    'status': 'Failed',
                    'actual_result': 'Field not found in dataset',
                    'details': f"Field '{field_name}' is not present in the dataset"
                })
                continue
            
            prompt = f"""
            Analyze this test case against the dataset sample and return ONLY a valid JSON object with these EXACT fields:
            - status: "Passed" or "Failed"
            - actual_result: Brief description of what happened
            - details: Explanation of the result
            
            Test Case:
            {json.dumps(test_case, indent=2)}
            
            Dataset Sample (first 5 rows):
            {dataset[field_name].head().to_dict()}
            
            IMPORTANT: Return ONLY the JSON object, no additional text or markdown formatting.
            The response must start with {{ and end with }}.
            """
            
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # More robust JSON extraction
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                # Find JSON object in response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")
                    
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                results.append({**test_case, **result})
                
            except Exception as e:
                results.append({
                    **test_case,
                    'status': 'Error',
                    'actual_result': 'Test execution failed',
                    'details': f"Error parsing AI response: {str(e)}. Raw response: {response_text if 'response_text' in locals() else 'N/A'}"
                })
        return pd.DataFrame(results)
        
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
    st.markdown("Upload your data dictionary and dataset to generate and execute test cases using AI")

    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Enter your Gemini API key
        2. Upload a CSV file with your data dictionary
        3. Upload your dataset CSV file
        4. Generate and execute test cases
        5. Review the test results
        """)

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📤 Upload Files")
        data_dict_file = st.file_uploader(
            "Upload Data Dictionary (CSV)",
            type=['csv'],
            help="Upload your data dictionary CSV file"
        )
        
        dataset_file = st.file_uploader(
            "Upload Dataset (CSV)",
            type=['csv'],
            help="Upload your dataset CSV file"
        )

    with col2:
        st.header("🚀 Test Case Execution")
        
        if data_dict_file and dataset_file and api_key:
            if st.button("Generate and Execute Test Cases", type="primary", use_container_width=True):
                try:
                    with st.spinner("🧠 AI is analyzing and executing test cases..."):
                        # Initialize generator
                        generator = DataDictionaryTestCaseGenerator(api_key)

                        # Load data dictionary
                        data_dict_file.seek(0)
                        data_dictionary = generator.load_data_dictionary_from_uploaded_file(data_dict_file)
                        
                        # Load dataset
                        dataset_file.seek(0)
                        dataset = pd.read_csv(dataset_file)
                        
                        # Generate test cases
                        test_cases = generator.generate_test_cases(data_dictionary)
                        
                        # Execute test cases
                        test_results = generator.execute_test_cases(test_cases, dataset)
                        
                        # Store in session state
                        st.session_state.test_results = test_results
                        st.session_state.data_dictionary = data_dictionary
                        st.session_state.dataset = dataset
                        
                        st.success("✅ Test execution completed!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            if not api_key:
                st.warning("⚠️ Please enter your Gemini API key")
            if not data_dict_file:
                st.warning("⚠️ Please upload a data dictionary file")
            if not dataset_file:
                st.warning("⚠️ Please upload a dataset file")

    # Display results if available
    if 'test_results' in st.session_state:
        st.header("📊 Test Results Dashboard")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Test Cases", len(st.session_state.test_results))
        with col2:
            passed = len(st.session_state.test_results[st.session_state.test_results['status'] == 'Passed'])
            st.metric("Passed Test Cases", passed)
        with col3:
            failed = len(st.session_state.test_results[st.session_state.test_results['status'] == 'Failed'])
            st.metric("Failed Test Cases", failed)

        # Test results table
        st.subheader("📋 Test Results Details")
        st.dataframe(st.session_state.test_results, use_container_width=True)

        # Download section
        st.subheader("💾 Download Test Results")
        csv_data = convert_df_to_csv(st.session_state.test_results)
        st.download_button(
            label="📥 Download Test Results (CSV)",
            data=csv_data,
            file_name="test_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Test results visualization
        st.subheader("📈 Test Results Distribution")
        status_counts = st.session_state.test_results['status'].value_counts()
        st.bar_chart(status_counts)

if __name__ == "__main__":
    main()