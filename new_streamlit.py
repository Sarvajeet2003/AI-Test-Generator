import streamlit as st
import pandas as pd
import numpy as np
import json
import google.generativeai as genai
from typing import Dict, List, Any
import io

class DataDictionaryTestCaseGenerator:
    def __init__(self, api_key: str):
        """Initialize the AI agent with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def load_data_dictionary_from_uploaded_file(self, uploaded_file) -> Dict:
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
    
    def load_dataset_from_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Load dataset from uploaded file"""
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def evaluate_dataset_with_ai(self, dataset: pd.DataFrame, test_cases: List[Dict], sample_size: int = None) -> Dict:
        """Evaluate dataset against test cases using AI with row-by-row analysis"""
        
        # If dataset is too large, sample it for AI analysis
        if sample_size and len(dataset) > sample_size:
            dataset_sample = dataset.sample(n=sample_size, random_state=42)
            st.info(f"Dataset is large ({len(dataset)} rows). Using random sample of {sample_size} rows for AI analysis.")
        else:
            dataset_sample = dataset
        
        # Convert dataset to a format suitable for AI analysis
        dataset_for_ai = self._prepare_dataset_for_ai(dataset_sample)
        
        prompt = f"""
        You are a data validation expert. Analyze the following dataset row by row against the provided test cases and generate a comprehensive evaluation report.
        
        Dataset to analyze (each row represents a record):
        {dataset_for_ai}
        
        Test Cases to apply on each row:
        {json.dumps(test_cases, indent=2)}
        
        Instructions:
        1. Examine each row of the dataset against each relevant test case
        2. For each test case, check ALL rows and identify which ones pass or fail
        3. Provide specific row numbers and values that fail validation
        4. Count the exact number of failures for each test case
        5. Provide concrete examples of failing data
        
        For each test case, analyze every row and determine:
        - Total rows that PASS the test case
        - Total rows that FAIL the test case
        - Specific row indices that fail (using 0-based indexing)
        - Actual failing values from those rows
        - Detailed explanation of why they fail
        
        Return the result as a JSON object with this structure:
        {{
            "overall_summary": {{
                "total_test_cases": number,
                "total_rows_analyzed": number,
                "passed_test_cases": number,
                "failed_test_cases": number,
                "overall_pass_rate": percentage
            }},
            "detailed_results": [
                {{
                    "test_case_id": "TC_001",
                    "field_name": "field_name",
                    "test_case_name": "test name",
                    "status": "PASS" or "FAIL",
                    "passed_rows": number,
                    "failed_rows": number,
                    "total_rows": number,
                    "row_level_pass_rate": percentage,
                    "failing_row_indices": [0, 1, 2],
                    "failing_values": ["value1", "value2", "value3"],
                    "failure_reason": "detailed explanation of why rows failed",
                    "failure_examples": [
                        {{
                            "row_index": 0,
                            "field_value": "actual_value",
                            "reason": "why this specific value failed"
                        }}
                    ],
                    "recommendations": "specific suggestions to fix the issues"
                }}
            ]
        }}
        
        Return only the JSON object, no explanations or markdown formatting.
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
            evaluation_result = json.loads(response_text)
            
            # If we used a sample, scale up the results
            if sample_size and len(dataset) > sample_size:
                evaluation_result = self._scale_results_to_full_dataset(evaluation_result, len(dataset), len(dataset_sample))
            
            return evaluation_result
            
        except Exception as e:
            st.error(f"AI evaluation failed: {e}")
            return self._generate_fallback_evaluation(dataset, test_cases)
    
    def _prepare_dataset_for_ai(self, dataset: pd.DataFrame) -> str:
        """Prepare dataset for AI analysis by converting to structured text format"""
        
        # Convert dataset to JSON with row indices for easy reference
        dataset_dict = []
        for idx, row in dataset.iterrows():
            row_dict = {"row_index": idx}
            for col in dataset.columns:
                # Handle NaN values
                value = row[col]
                if pd.isna(value):
                    value = "NULL"
                elif isinstance(value, (int, float)):
                    value = str(value)
                else:
                    value = str(value)
                row_dict[col] = value
            dataset_dict.append(row_dict)
        
        return json.dumps(dataset_dict, indent=2, default=str)
    
    def _scale_results_to_full_dataset(self, evaluation_result: Dict, full_size: int, sample_size: int) -> Dict:
        """Scale the evaluation results from sample to full dataset"""
        scaling_factor = full_size / sample_size
        
        # Update overall summary
        evaluation_result["overall_summary"]["total_rows_analyzed"] = full_size
        
        # Scale detailed results
        for result in evaluation_result["detailed_results"]:
            result["passed_rows"] = int(result["passed_rows"] * scaling_factor)
            result["failed_rows"] = int(result["failed_rows"] * scaling_factor)
            result["total_rows"] = full_size
            
            # Recalculate pass rate
            if full_size > 0:
                result["row_level_pass_rate"] = (result["passed_rows"] / full_size) * 100
        
        return evaluation_result
    
    def evaluate_dataset_in_chunks(self, dataset: pd.DataFrame, test_cases: List[Dict], chunk_size: int = 100) -> Dict:
        """Evaluate large datasets by processing in chunks"""
        
        total_rows = len(dataset)
        chunks = [dataset[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        
        st.info(f"Processing large dataset in {len(chunks)} chunks of {chunk_size} rows each...")
        
        # Initialize results structure
        aggregated_results = {
            "overall_summary": {
                "total_test_cases": len(test_cases),
                "total_rows_analyzed": total_rows,
                "passed_test_cases": 0,
                "failed_test_cases": 0,
                "overall_pass_rate": 0
            },
            "detailed_results": []
        }
        
        # Initialize detailed results structure
        for test_case in test_cases:
            aggregated_results["detailed_results"].append({
                "test_case_id": test_case["test_case_id"],
                "field_name": test_case["field_name"],
                "test_case_name": test_case["test_case_name"],
                "status": "PASS",
                "passed_rows": 0,
                "failed_rows": 0,
                "total_rows": total_rows,
                "row_level_pass_rate": 0,
                "failing_row_indices": [],
                "failing_values": [],
                "failure_reason": "",
                "failure_examples": [],
                "recommendations": ""
            })
        
        # Process each chunk
        progress_bar = st.progress(0)
        for chunk_idx, chunk in enumerate(chunks):
            progress_bar.progress((chunk_idx + 1) / len(chunks))
            
            try:
                # Evaluate current chunk
                chunk_result = self.evaluate_dataset_with_ai(chunk, test_cases)
                
                # Aggregate results
                for i, test_result in enumerate(chunk_result["detailed_results"]):
                    agg_result = aggregated_results["detailed_results"][i]
                    
                    # Add to totals
                    agg_result["passed_rows"] += test_result["passed_rows"]
                    agg_result["failed_rows"] += test_result["failed_rows"]
                    
                    # Adjust row indices for the chunk offset
                    chunk_offset = chunk_idx * chunk_size
                    adjusted_indices = [idx + chunk_offset for idx in test_result.get("failing_row_indices", [])]
                    agg_result["failing_row_indices"].extend(adjusted_indices)
                    
                    # Collect failing values and examples
                    agg_result["failing_values"].extend(test_result.get("failing_values", []))
                    
                    # Keep examples from different chunks
                    for example in test_result.get("failure_examples", []):
                        example["row_index"] += chunk_offset
                        agg_result["failure_examples"].append(example)
                    
                    # Update failure reason and recommendations (keep the most recent)
                    if test_result.get("failure_reason"):
                        agg_result["failure_reason"] = test_result["failure_reason"]
                    if test_result.get("recommendations"):
                        agg_result["recommendations"] = test_result["recommendations"]
            
            except Exception as e:
                st.warning(f"Error processing chunk {chunk_idx + 1}: {e}")
                continue
        
        # Calculate final statistics
        for result in aggregated_results["detailed_results"]:
            if result["failed_rows"] > 0:
                result["status"] = "FAIL"
                aggregated_results["overall_summary"]["failed_test_cases"] += 1
            else:
                aggregated_results["overall_summary"]["passed_test_cases"] += 1
            
            # Calculate pass rate
            if result["total_rows"] > 0:
                result["row_level_pass_rate"] = (result["passed_rows"] / result["total_rows"]) * 100
            
            # Limit the number of examples and failing values to avoid overwhelming output
            result["failure_examples"] = result["failure_examples"][:10]  # Keep first 10 examples
            result["failing_values"] = list(set(result["failing_values"]))[:20]  # Keep unique values, max 20
        
        # Calculate overall pass rate
        if aggregated_results["overall_summary"]["total_test_cases"] > 0:
            aggregated_results["overall_summary"]["overall_pass_rate"] = (
                aggregated_results["overall_summary"]["passed_test_cases"] / 
                aggregated_results["overall_summary"]["total_test_cases"]
            ) * 100
        
        progress_bar.empty()
        return aggregated_results
    
    def _generate_fallback_evaluation(self, dataset: pd.DataFrame, test_cases: List[Dict]) -> Dict:
        """Generate basic evaluation when AI fails"""
        passed = 0
        failed = 0
        detailed_results = []
        
        for test_case in test_cases:
            field_name = test_case['field_name']
            
            # Basic validation
            if field_name in dataset.columns:
                status = "PASS"
                reason = f"Field {field_name} exists in dataset"
                passed += 1
                passed_rows = len(dataset)
                failed_rows = 0
            else:
                status = "FAIL"
                reason = f"Field {field_name} not found in dataset"
                failed += 1
                passed_rows = 0
                failed_rows = len(dataset)
            
            detailed_results.append({
                "test_case_id": test_case['test_case_id'],
                "field_name": field_name,
                "test_case_name": test_case['test_case_name'],
                "status": status,
                "passed_rows": passed_rows,
                "failed_rows": failed_rows,
                "total_rows": len(dataset),
                "row_level_pass_rate": (passed_rows / len(dataset)) * 100 if len(dataset) > 0 else 0,
                "failing_row_indices": list(range(len(dataset))) if status == "FAIL" else [],
                "failing_values": [],
                "failure_reason": reason,
                "failure_examples": [],
                "recommendations": "Check data dictionary alignment" if status == "FAIL" else "No issues found"
            })
        
        return {
            "overall_summary": {
                "total_test_cases": len(test_cases),
                "total_rows_analyzed": len(dataset),
                "passed_test_cases": passed,
                "failed_test_cases": failed,
                "overall_pass_rate": (passed / len(test_cases)) * 100 if test_cases else 0
            },
            "detailed_results": detailed_results
        }

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.set_page_config(
        page_title="Test Case Generator Dashboard",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Data Dictionary Test Case Generator & Validator")
    st.markdown("Upload your data dictionary and dataset to generate comprehensive test cases and validation reports using AI")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        
        st.divider()
        st.header("🔧 Processing Options")
        
        # Dataset processing options
        processing_mode = st.radio(
            "Dataset Processing Mode:",
            ["Smart Sampling", "Chunk Processing", "Full Dataset"],
            help="Choose how to process your dataset"
        )
        
        if processing_mode == "Smart Sampling":
            sample_size = st.slider("Sample Size", min_value=50, max_value=1000, value=200, step=50)
        elif processing_mode == "Chunk Processing":
            chunk_size = st.slider("Chunk Size", min_value=50, max_value=500, value=100, step=50)
        
        st.divider()
        st.header("📊 Process Flow")
        st.markdown("""
        1. **Upload Data Dictionary** (CSV)
        2. **Generate Test Cases** using AI
        3. **Upload Dataset** (CSV/Excel)
        4. **AI Row-by-Row Validation**
        5. **Review Detailed Results**
        6. **Download Comprehensive Reports**
        """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Generate", "🔍 Validate Dataset", "📊 Results Dashboard"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📤 Upload Data Dictionary")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your data dictionary CSV file",
                key="data_dict_uploader"
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
                            st.session_state.generator = generator
                            
                            st.success(f"✅ Generated {len(test_cases)} test cases")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                if not api_key:
                    st.warning("⚠️ Please enter your Gemini API key")
                if not uploaded_file:
                    st.warning("⚠️ Please upload a data dictionary file")
    
    with tab2:
        st.header("🔍 Dataset Validation")
        
        if 'test_cases' not in st.session_state:
            st.warning("⚠️ Please generate test cases first in the 'Upload & Generate' tab")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Upload Dataset")
                dataset_file = st.file_uploader(
                    "Choose your dataset file",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload the dataset you want to validate",
                    key="dataset_uploader"
                )
                
                if dataset_file is not None:
                    st.success(f"Dataset uploaded: {dataset_file.name}")
                    
                    try:
                        # Load dataset
                        dataset = st.session_state.generator.load_dataset_from_uploaded_file(dataset_file)
                        st.session_state.dataset = dataset
                        
                        # Display dataset preview
                        st.subheader("📋 Dataset Preview")
                        st.dataframe(dataset.head(10), use_container_width=True)
                        
                        # Dataset info
                        st.info(f"Dataset shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
                        
                        # Show processing recommendation
                        if len(dataset) > 1000:
                            st.warning(f"⚠️ Large dataset detected ({len(dataset)} rows). Consider using 'Smart Sampling' or 'Chunk Processing' mode for better performance.")
                        
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
            
            with col2:
                st.subheader("🧪 Run AI Validation")
                
                if 'dataset' in st.session_state:
                    if st.button("🚀 Run Row-by-Row AI Validation", type="primary", use_container_width=True):
                        try:
                            dataset = st.session_state.dataset
                            test_cases = st.session_state.test_cases
                            
                            with st.spinner("🤖 AI is performing row-by-row validation..."):
                                # Choose processing method based on user selection
                                if processing_mode == "Smart Sampling":
                                    evaluation_result = st.session_state.generator.evaluate_dataset_with_ai(
                                        dataset, test_cases, sample_size=sample_size
                                    )
                                elif processing_mode == "Chunk Processing":
                                    evaluation_result = st.session_state.generator.evaluate_dataset_in_chunks(
                                        dataset, test_cases, chunk_size=chunk_size
                                    )
                                else:  # Full Dataset
                                    if len(dataset) > 500:
                                        st.warning("Processing full dataset. This may take a while...")
                                    evaluation_result = st.session_state.generator.evaluate_dataset_with_ai(
                                        dataset, test_cases
                                    )
                                
                                # Store results
                                st.session_state.evaluation_result = evaluation_result
                                
                                # Show quick summary
                                summary = evaluation_result['overall_summary']
                                st.success(f"✅ Row-by-row validation completed!")
                                
                                # Quick metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Tests", summary['total_test_cases'])
                                with col2:
                                    st.metric("Passed Tests", summary['passed_test_cases'])
                                with col3:
                                    st.metric("Failed Tests", summary['failed_test_cases'])
                                
                                st.info(f"Overall Pass Rate: {summary['overall_pass_rate']:.1f}%")
                                st.info(f"Rows Analyzed: {summary['total_rows_analyzed']:,}")
                        
                        except Exception as e:
                            st.error(f"❌ Validation failed: {e}")
                else:
                    st.warning("⚠️ Please upload a dataset first")
    
    with tab3:
        st.header("📊 Detailed Validation Results")
        
        if 'evaluation_result' not in st.session_state:
            st.warning("⚠️ Please run validation first in the 'Validate Dataset' tab")
        else:
            evaluation_result = st.session_state.evaluation_result
            
            # Overall Summary
            st.subheader("📈 Overall Summary")
            summary = evaluation_result['overall_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Test Cases", summary['total_test_cases'])
            with col2:
                st.metric("Passed Tests", summary['passed_test_cases'], delta=f"+{summary['passed_test_cases']}")
            with col3:
                st.metric("Failed Tests", summary['failed_test_cases'], delta=f"-{summary['failed_test_cases']}")
            with col4:
                st.metric("Rows Analyzed", f"{summary['total_rows_analyzed']:,}")
            
            # Progress bar for overall pass rate
            st.write("**Overall Test Pass Rate:**")
            st.progress(summary['overall_pass_rate'] / 100)
            st.write(f"{summary['overall_pass_rate']:.1f}%")
            
            # Detailed Results
            st.subheader("📋 Row-by-Row Test Results")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox(
                    "Filter by Status:",
                    options=["All", "PASS", "FAIL"]
                )
            with col2:
                field_filter = st.selectbox(
                    "Filter by Field:",
                    options=["All"] + list(set([result['field_name'] for result in evaluation_result['detailed_results']]))
                )
            
            # Filter results
            filtered_results = evaluation_result['detailed_results']
            if status_filter != "All":
                filtered_results = [r for r in filtered_results if r['status'] == status_filter]
            if field_filter != "All":
                filtered_results = [r for r in filtered_results if r['field_name'] == field_filter]
            
            # Display detailed results
            for result in filtered_results:
                status_icon = "❌" if result['status'] == 'FAIL' else "✅"
                row_pass_rate = result.get('row_level_pass_rate', 0)
                
                with st.expander(f"{status_icon} {result['test_case_name']} - {result['status']} ({row_pass_rate:.1f}% rows passed)", expanded=result['status'] == 'FAIL'):
                    
                    # Test case info
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Field:** {result['field_name']}")
                        st.write(f"**Test Case ID:** {result['test_case_id']}")
                        st.write(f"**Status:** {result['status']}")
                    
                    with col2:
                        # Row-level metrics
                        st.metric("Passed Rows", f"{result['passed_rows']:,}")
                        st.metric("Failed Rows", f"{result['failed_rows']:,}")
                        st.metric("Row Pass Rate", f"{row_pass_rate:.1f}%")
                    
                    # Progress bar for row pass rate
                    st.progress(row_pass_rate / 100)
                    
                    if result['status'] == 'FAIL':
                        st.write(f"**Failure Reason:** {result.get('failure_reason', 'Not specified')}")
                        
                        # Show failing row indices (limited)
                        failing_indices = result.get('failing_row_indices', [])
                        if failing_indices:
                            st.write(f"**Sample Failing Row Indices:** {failing_indices[:10]} {'...' if len(failing_indices) > 10 else ''}")
                        
                        # Show failing values (unique, limited)
                        failing_values = result.get('failing_values', [])
                        if failing_values:
                            unique_failing_values = list(set(failing_values))[:10]
                            st.write("**Sample Failing Values:**")
                            for value in unique_failing_values:
                                st.code(str(value))
                        
                        # Show specific failure examples
                        failure_examples = result.get('failure_examples', [])
                        if failure_examples:
                            st.write("**Specific Failure Examples:**")
                            for i, example in enumerate(failure_examples[:5]):  # Show first 5 examples
                                st.write(f"• Row {example.get('row_index', 'N/A')}: `{example.get('field_value', 'N/A')}` - {example.get('reason', 'No reason provided')}")
                        
                        # Recommendations
                        recommendations = result.get('recommendations', '')
                        if recommendations:
                            st.write("**Recommendations:**")
                            st.info(recommendations)
            
                        # Export options
            st.subheader("📥 Export Reports")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Detailed report with all results
                detailed_data = []
                for result in evaluation_result['detailed_results']:
                    detailed_data.append({
                        'Test Case ID': result['test_case_id'],
                        'Field Name': result['field_name'],
                        'Test Case Name': result['test_case_name'],
                        'Status': result['status'],
                        'Passed Rows': result['passed_rows'],
                        'Failed Rows': result['failed_rows'],
                        'Total Rows': result['total_rows'],
                        'Row Pass Rate (%)': round(result.get('row_level_pass_rate', 0), 2),
                        'Failure Reason': result.get('failure_reason', ''),
                        'Sample Failing Values': '; '.join(map(str, list(set(result.get('failing_values', [])))[:5])),
                        'Recommendations': result.get('recommendations', '')
                    })
                
                detailed_df = pd.DataFrame(detailed_data)
                csv_detailed = convert_df_to_csv(detailed_df)
                
                st.download_button(
                    label="📄 Download Detailed Report",
                    data=csv_detailed,
                    file_name=f"detailed_validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Summary report
                summary_data = {
                    'Metric': ['Total Test Cases', 'Passed Test Cases', 'Failed Test Cases', 'Total Rows Analyzed', 'Overall Pass Rate (%)'],
                    'Value': [
                        summary['total_test_cases'],
                        summary['passed_test_cases'],
                        summary['failed_test_cases'],
                        summary['total_rows_analyzed'],
                        round(summary['overall_pass_rate'], 2)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv_summary = convert_df_to_csv(summary_df)
                
                st.download_button(
                    label="📊 Download Summary Report",
                    data=csv_summary,
                    file_name=f"validation_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Row-level failure report
                if any(result['status'] == 'FAIL' for result in evaluation_result['detailed_results']):
                    failure_data = []
                    for result in evaluation_result['detailed_results']:
                        if result['status'] == 'FAIL':
                            failing_indices = result.get('failing_row_indices', [])
                            for idx in failing_indices[:100]:  # Limit to first 100 failing rows per test
                                failure_data.append({
                                    'Test Case ID': result['test_case_id'],
                                    'Field Name': result['field_name'],
                                    'Row Index': idx,
                                    'Failure Reason': result.get('failure_reason', '')
                                })
                    
                    if failure_data:
                        failure_df = pd.DataFrame(failure_data)
                        csv_failures = convert_df_to_csv(failure_df)
                        
                        st.download_button(
                            label="🚨 Download Failure Details",
                            data=csv_failures,
                            file_name=f"row_level_failures_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No failures to export!")
                else:
                    st.success("🎉 No failures to export - All tests passed!")
            
            # Data Quality Score
            st.subheader("🏆 Data Quality Score")
            quality_score = summary['overall_pass_rate']
            
            if quality_score >= 90:
                score_color = "green"
                score_emoji = "🏆"
                score_text = "Excellent"
            elif quality_score >= 75:
                score_color = "blue"
                score_emoji = "🥉"
                score_text = "Good"
            elif quality_score >= 50:
                score_color = "orange"
                score_emoji = "⚠️"
                score_text = "Needs Improvement"
            else:
                score_color = "red"
                score_emoji = "🚨"
                score_text = "Poor"
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### {score_emoji} {quality_score:.1f}/100")
                st.markdown(f"**{score_text}**")
            
            with col2:
                st.markdown("**Quality Assessment:**")
                if quality_score >= 90:
                    st.success("Your data quality is excellent! Most validation rules are satisfied.")
                elif quality_score >= 75:
                    st.info("Your data quality is good, but there are some issues to address.")
                elif quality_score >= 50:
                    st.warning("Your data quality needs improvement. Several validation rules failed.")
                else:
                    st.error("Your data quality is poor. Many validation rules failed and immediate attention is required.")
            
            # Failure Analysis
            if summary['failed_test_cases'] > 0:
                st.subheader("🔍 Failure Analysis")
                
                # Group failures by type
                failure_types = {}
                field_failures = {}
                
                for result in evaluation_result['detailed_results']:
                    if result['status'] == 'FAIL':
                        # Get test type from test cases
                        test_type = next((tc.get('test_type', 'unknown') for tc in st.session_state.test_cases 
                                        if tc['test_case_id'] == result['test_case_id']), 'unknown')
                        
                        if test_type not in failure_types:
                            failure_types[test_type] = 0
                        failure_types[test_type] += 1
                        
                        field_name = result['field_name']
                        if field_name not in field_failures:
                            field_failures[field_name] = 0
                        field_failures[field_name] += 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Failures by Test Type:**")
                    for test_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• {test_type}: {count} failures")
                
                with col2:
                    st.write("**Failures by Field:**")
                    for field, count in sorted(field_failures.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• {field}: {count} failures")
                
                # Recommendations
                st.subheader("💡 Overall Recommendations")
                
                recommendations = []
                if failure_types.get('data_type', 0) > 0:
                    recommendations.append("🔧 **Data Type Issues**: Review data types and ensure proper conversion during data ingestion.")
                
                if failure_types.get('required', 0) > 0:
                    recommendations.append("📋 **Missing Required Fields**: Implement data validation at the source to prevent missing required values.")
                
                if failure_types.get('format', 0) > 0:
                    recommendations.append("📝 **Format Issues**: Standardize data formats and implement format validation rules.")
                
                if failure_types.get('range', 0) > 0:
                    recommendations.append("📊 **Range Violations**: Review business rules and implement range checks during data entry.")
                
                if failure_types.get('allowed_values', 0) > 0:
                    recommendations.append("📚 **Invalid Values**: Create reference data validation and dropdown lists to prevent invalid entries.")
                
                if not recommendations:
                    recommendations.append("✅ **General**: Continue monitoring data quality and run regular validations.")
                
                for rec in recommendations:
                    st.markdown(rec)
    
    # Display test cases if available
    if 'test_cases' in st.session_state:
        with st.expander("🧪 View Generated Test Cases", expanded=False):
            st.header("📋 Generated Test Cases")
            
            # Test cases metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Test Cases", len(st.session_state.test_cases))
            with col2:
                test_types = [tc.get('test_type', 'unknown') for tc in st.session_state.test_cases]
                st.metric("Unique Test Types", len(set(test_types)))
            with col3:
                fields = [tc['field_name'] for tc in st.session_state.test_cases]
                st.metric("Fields Covered", len(set(fields)))
            
            # Display test cases table
            df_test_cases = pd.DataFrame(st.session_state.test_cases)
            
            # Filter options for test cases
            col1, col2 = st.columns(2)
            with col1:
                tc_field_filter = st.multiselect(
                    "Filter by Field:",
                    options=df_test_cases['field_name'].unique(),
                    default=df_test_cases['field_name'].unique(),
                    key="tc_field_filter"
                )
            with col2:
                tc_type_filter = st.multiselect(
                    "Filter by Test Type:",
                    options=df_test_cases['test_type'].unique() if 'test_type' in df_test_cases.columns else [],
                    default=df_test_cases['test_type'].unique() if 'test_type' in df_test_cases.columns else [],
                    key="tc_type_filter"
                )
            
            # Filter dataframe
            filtered_tc_df = df_test_cases[
                (df_test_cases['field_name'].isin(tc_field_filter)) &
                (df_test_cases['test_type'].isin(tc_type_filter) if 'test_type' in df_test_cases.columns else True)
            ]
            
            st.dataframe(filtered_tc_df, use_container_width=True, hide_index=True)
            
            # Download test cases
            csv_test_cases = convert_df_to_csv(df_test_cases)
            st.download_button(
                label="📥 Download Test Cases",
                data=csv_test_cases,
                file_name=f"generated_test_cases_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()