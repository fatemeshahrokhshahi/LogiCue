import json
import os
import pandas as pd
from pathlib import Path
import re

class LogiCueErrorAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompting methods to analyze
        self.prompting_methods = ['zero-shot', 'chain-of-thought', 'LogiCue']
        
        # Inference patterns (excluding NSFC)
        self.inference_patterns = ['DSmu', 'DSmi', 'MTmu', 'MTmi', 'CT', 'CMP', 'ASd']
        
        # Store all data
        self.all_responses = []
        self.error_cases = []
        
    def extract_answer(self, response_text):
        """Extract yes/no answer from response text"""
        response_lower = response_text.lower().strip()
        
        # Look for "Answer: yes" or "Answer: no" pattern
        answer_pattern = r'answer:\s*(yes|no)'
        match = re.search(answer_pattern, response_lower)
        if match:
            return match.group(1)
        
        # Look for standalone yes/no at the end
        if response_lower.endswith('yes'):
            return 'yes'
        elif response_lower.endswith('no'):
            return 'no'
        
        # Look for yes/no anywhere in short responses
        if len(response_lower) < 50:
            if 'yes' in response_lower and 'no' not in response_lower:
                return 'yes'
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 'no'
        
        return 'unclear'
    
    def read_json_files(self):
        """Read all JSON files from the specified prompting methods"""
        print("Reading JSON files...")
        
        for prompting_method in self.prompting_methods:
            method_path = self.data_dir / prompting_method
            
            if not method_path.exists():
                print(f"Warning: {method_path} does not exist. Skipping...")
                continue
            
            # Iterate through all model directories
            for model_dir in method_path.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                
                # Iterate through inference pattern directories
                for pattern_dir in model_dir.iterdir():
                    if not pattern_dir.is_dir():
                        continue
                    
                    inference_pattern = pattern_dir.name
                    
                    # Skip NSFC pattern
                    if inference_pattern == 'NSFC':
                        continue
                    
                    # Skip if not in our list of patterns
                    if inference_pattern not in self.inference_patterns:
                        continue
                    
                    # Read all JSON files in this directory
                    json_files = list(pattern_dir.glob('*.json'))
                    
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Extract question number from filename
                            # e.g., claude_haiku_4_5_20251001_DSmu_1_0.json -> 1
                            filename = json_file.stem
                            parts = filename.split('_')
                            question_num = None
                            for i, part in enumerate(parts):
                                if part == inference_pattern and i + 1 < len(parts):
                                    question_num = parts[i + 1]
                                    break
                            
                            # Extract response content
                            response_text = ""
                            if 'responses' in data and len(data['responses']) > 0:
                                response_text = data['responses'][0].get('content', '')
                            
                            # Extract answer
                            answer = self.extract_answer(response_text)
                            
                            # Store the data
                            record = {
                                'prompting_method': prompting_method,
                                'model': model_name,
                                'inference_pattern': inference_pattern,
                                'question_num': question_num,
                                'question': data.get('user_prompt', ''),
                                'system_prompt': data.get('system_prompt', ''),
                                'response': response_text,
                                'answer': answer,
                                'is_correct': answer == 'no',
                                'file_path': str(json_file)
                            }
                            
                            self.all_responses.append(record)
                            
                            # If incorrect (answer is yes), add to error cases
                            if answer == 'yes':
                                self.error_cases.append(record)
                        
                        except Exception as e:
                            print(f"Error reading {json_file}: {e}")
        
        print(f"Total responses read: {len(self.all_responses)}")
        print(f"Total error cases (answered 'yes'): {len(self.error_cases)}")
    
    def save_error_cases_for_analysis(self):
        """Save error cases in a readable format for manual analysis"""
        
        # Group by inference pattern
        for pattern in self.inference_patterns:
            pattern_errors = [e for e in self.error_cases if e['inference_pattern'] == pattern]
            
            if not pattern_errors:
                continue
            
            # Create output file for this pattern
            output_file = self.output_dir / f"{pattern}_errors_for_analysis.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"="*100 + "\n")
                f.write(f"ERROR ANALYSIS FOR PATTERN: {pattern}\n")
                f.write(f"Total Errors: {len(pattern_errors)}\n")
                f.write(f"="*100 + "\n\n")
                
                # Group by prompting method
                for method in self.prompting_methods:
                    method_errors = [e for e in pattern_errors if e['prompting_method'] == method]
                    
                    if not method_errors:
                        continue
                    
                    f.write(f"\n{'='*100}\n")
                    f.write(f"PROMPTING METHOD: {method.upper()}\n")
                    f.write(f"Errors in this method: {len(method_errors)}\n")
                    f.write(f"{'='*100}\n\n")
                    
                    # Group by model
                    models = sorted(set(e['model'] for e in method_errors))
                    
                    for model in models:
                        model_errors = [e for e in method_errors if e['model'] == model]
                        
                        f.write(f"\n{'-'*100}\n")
                        f.write(f"MODEL: {model}\n")
                        f.write(f"Errors: {len(model_errors)}\n")
                        f.write(f"{'-'*100}\n\n")
                        
                        for idx, error in enumerate(model_errors, 1):
                            f.write(f"\nERROR #{idx} (Question {error['question_num']})\n")
                            f.write(f"{'-'*80}\n")
                            f.write(f"QUESTION:\n{error['question']}\n\n")
                            
                            if error['system_prompt']:
                                f.write(f"SYSTEM PROMPT:\n{error['system_prompt']}\n\n")
                            
                            f.write(f"MODEL RESPONSE:\n{error['response']}\n\n")
                            f.write(f"ANSWER GIVEN: {error['answer']}\n")
                            f.write(f"CORRECT ANSWER: no\n")
                            f.write(f"\nFILE: {error['file_path']}\n")
                            f.write(f"\n{'='*80}\n\n")
            
            print(f"Saved error analysis for {pattern} to {output_file}")
    
    def create_summary_statistics(self):
        """Create summary statistics of errors"""
        
        df = pd.DataFrame(self.all_responses)
        
        # Overall accuracy by prompting method
        accuracy_by_method = df.groupby('prompting_method').agg({
            'is_correct': ['sum', 'count', 'mean']
        }).round(4)
        accuracy_by_method.columns = ['Correct', 'Total', 'Accuracy']
        
        # Accuracy by pattern and method
        accuracy_by_pattern_method = df.groupby(['inference_pattern', 'prompting_method']).agg({
            'is_correct': ['sum', 'count', 'mean']
        }).round(4)
        accuracy_by_pattern_method.columns = ['Correct', 'Total', 'Accuracy']
        
        # Accuracy by model and pattern
        accuracy_by_model_pattern = df.groupby(['model', 'inference_pattern', 'prompting_method']).agg({
            'is_correct': ['sum', 'count', 'mean']
        }).round(4)
        accuracy_by_model_pattern.columns = ['Correct', 'Total', 'Accuracy']
        
        # Save to Excel
        excel_path = self.output_dir / 'error_summary_statistics.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            accuracy_by_method.to_excel(writer, sheet_name='By Method')
            accuracy_by_pattern_method.to_excel(writer, sheet_name='By Pattern & Method')
            accuracy_by_model_pattern.to_excel(writer, sheet_name='By Model & Pattern')
            
            # Error counts by pattern
            error_df = pd.DataFrame(self.error_cases)
            if not error_df.empty:
                error_summary = error_df.groupby(['inference_pattern', 'prompting_method', 'model']).size().reset_index(name='Error Count')
                error_summary.to_excel(writer, sheet_name='Error Counts', index=False)
        
        print(f"Saved summary statistics to {excel_path}")
        
        return df
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting LogiCue Error Analysis...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Read all JSON files
        self.read_json_files()
        
        # Save error cases for manual analysis
        print("\nSaving error cases for manual analysis...")
        self.save_error_cases_for_analysis()
        
        # Create summary statistics
        print("\nCreating summary statistics...")
        df = self.create_summary_statistics()
        
        print("\n" + "="*100)
        print("ANALYSIS COMPLETE!")
        print("="*100)
        print(f"\nCheck the '{self.output_dir}' folder for:")
        print(f"1. Individual error analysis files for each pattern (*_errors_for_analysis.txt)")
        print(f"2. Summary statistics (error_summary_statistics.xlsx)")
        print("\nYou can now manually review the error cases and classify them based on:")
        print("  - Necessity denial vs. definitive negation confusion (MTmu, MTmi)")
        print("  - Possibility vs. negation confusion (DSmi, MTmi)")
        print("  - Modal claim vs. content confusion (DSmu)")
        print("  - Material conditional vs. natural language conditional (CT)")
        print("  - Probability preservation issues (CMP)")
        print("  - Defeasibility issues (ASd)")
        
        return df

# Run the analysis
if __name__ == "__main__":
    data_dir = r""
    output_dir = r""
    
    analyzer = LogiCueErrorAnalyzer(data_dir, output_dir)
    df = analyzer.run_analysis()