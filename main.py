import google.generativeai as genai
import json
import pandas as pd
import csv
from typing import Dict, List, Union
import os

class DataDictionaryTestCaseGenerator:
    """
    AI Agent that reads data dictionary from file and generates test cases using Gemini-1.5-flash
    """
    
    def __init__(self, api_key: str):
        """Initialize the AI agent with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def load_data_dictionary(self, file_path: str) -> Dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data dictionary file not found: {file_path}")
        
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'csv':
                return self._load_from_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise Exception(f"Error loading data dictionary: {str(e)}")
    

    
    def _load_from_csv(self, file_path: str) -> Dict:
        """Load data dictionary from CSV file"""
        df = pd.read_csv(file_path)
        
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
        
        print(f"✅ Loaded data dictionary from CSV: {file_path}")
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
            print(f"AI generation failed: {e}")
            print("Using fallback test case generation...")
            return self.generate_fallback_test_cases(data_dictionary)
    
    def save_test_cases_csv(self, test_cases: List[Dict], filename: str = "test_cases.csv"):
        """Save test cases to CSV file"""
        df = pd.DataFrame(test_cases)
        df.to_csv(filename, index=False)
        print(f"✅ Test cases saved to {filename}")
    
    def display_test_cases(self, test_cases: List[Dict]):
        """Display test cases in a formatted way"""
        print("\n" + "="*80)
        print("GENERATED TEST CASES")
        print("="*80)
        
        for tc in test_cases:
            print(f"\n📋 {tc['test_case_id']}: {tc['test_case_name']}")
            print(f"   Field: {tc['field_name']}")
            print(f"   Type: {tc['test_type']}")
            print(f"   Description: {tc['test_description']}")
            print(f"   Rule: {tc['validation_rule']}")
            print(f"   Expected: {tc['expected_result']}")
            print("-" * 80)
        
        print(f"\n📊 Total Test Cases Generated: {len(test_cases)}")


def main():
    """Main function to demonstrate the test case generator"""
    
    # Replace with your actual Gemini API key
    API_KEY = "AIzaSyCEd6f7y1XTUG4P42tEwSdT1_Nf-h76sRs"
    
    # Create sample files for testing (remove this in production)
    # create_sample_data_dictionary_files()
    
    print("🤖 Starting Data Dictionary Test Case Generator...")
    
    # Initialize the generator
    generator = DataDictionaryTestCaseGenerator(API_KEY)
    
    # Load data dictionary from file
    data_dict_file = "data_dictionary.json"  # Change this to your file path
    # Other supported formats: "data_dictionary.csv", "data_dictionary.xlsx"
    
    print(f"📖 Loading data dictionary from: {data_dict_file}")
    data_dictionary = generator.load_data_dictionary(data_dict_file)
    
    print(f"📊 Found {len(data_dictionary)} fields in data dictionary")
    
    # Generate test cases
    print("🔄 Analyzing data dictionary and generating test cases...")
    test_cases = generator.generate_test_cases(data_dictionary)
    
    # Display test cases
    generator.display_test_cases(test_cases)
    
    # Save to both JSON and CSV
    generator.save_test_cases_csv(test_cases, "generated_test_cases.csv")
    
    print("\n✨ Test case generation completed!")
    print(f"📁 Files created:")
    print(f"   - generated_test_cases.json")
    print(f"   - generated_test_cases.csv")

if __name__ == "__main__":
    main()