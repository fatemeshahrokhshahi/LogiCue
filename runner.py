"""
LogiCue: Prompt Runner for Modal and Conditional Reasoning Experiments

Usage:
    1. Copy config.example.py to config.py
    2. Add your API keys to config.py
    3. Edit the __main__ section below with your experiment configuration
    4. Run: python runner.py
"""

import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import re

# Import configuration
try:
    from config import API_KEYS, PATHS
except ImportError:
    print("❌ Config file not found!")
    print("Please copy config.example.py to config.py and add your API keys.")
    exit(1)


class Runner:
    def __init__(self, 
                 repo_path: str = None,
                 ollama_url: str = "http://localhost:11434",
                 gemini_api_key: str = None,
                 deepseek_api_key: str = None,
                 anthropic_api_key: str = None,
                 openai_api_key: str = None):
        """
        Initialize the LogiCue Prompt Runner
        
        Args:
            repo_path: Path to your LogiCue repository (default: current directory)
            ollama_url: URL where Ollama is running
            gemini_api_key: Google Gemini API key
            deepseek_api_key: DeepSeek API key
            anthropic_api_key: Anthropic API key
            openai_api_key: OpenAI API key
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.ollama_url = ollama_url
        self.gemini_api_key = gemini_api_key
        self.deepseek_api_key = deepseek_api_key
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        
    def load_prompts(self, json_file_path: str) -> List[tuple]:
        """Load prompts from a JSON file"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                print(f"Error fetching models: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running!")
            return []
    
    def unload_all_models(self):
        """Force unload all models to free maximum memory"""
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                loaded = response.json().get('models', [])
                for model_info in loaded:
                    model_name = model_info.get('name', '')
                    if model_name:
                        self.unload_model(model_name)
            
            url = f"{self.ollama_url}/api/generate"
            requests.post(url, json={"model": "", "keep_alive": 0}, timeout=5)
            print(f"🧹 Cleared all models from memory")
        except Exception as e:
            print(f"⚠️  Could not unload models: {e}")
    
    def unload_model(self, model: str):
        """Unload a model from memory to free up resources"""
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {"model": model, "keep_alive": 0}
            requests.post(url, json=payload, timeout=10)
        except:
            pass
    
    def query_openai(self, 
                     model: str,
                     user_prompt: str,
                     system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                     temperature: float = 0,
                     max_retries: int = 5) -> Dict[str, Any]:
        """Query OpenAI API"""
        if not self.openai_api_key:
            print("❌ OpenAI API key not provided!")
            return None
        
        url = "https://api.openai.com/v1/responses"
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
    
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
        
        payload = {
            "model": model,
            "input": full_prompt,
            "temperature": temperature
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["output"][0]["content"][0]["text"]
                    return {"message": {"content": text}}
                
                elif response.status_code == 429:
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    retry_after = response.headers.get("Retry-After", "60")
                    try:
                        wait_time = int(retry_after)
                    except:
                        wait_time = 60
                    time.sleep(wait_time)
                    continue
                
                else:
                    print(f"❌ OpenAI Error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
            
            except Exception as e:
                print(f"❌ Error querying OpenAI: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
        
        return None
    
    def query_deepseek(self, 
                       model: str,
                       user_prompt: str,
                       system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                       temperature: float = 0,
                       max_retries: int = 10) -> Dict[str, Any]:
        """Query DeepSeek API"""
        if not self.deepseek_api_key:
            print("❌ DeepSeek API key not provided!")
            return None
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 8000,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=180)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"]
                    return {"message": {"content": text}}
                
                elif response.status_code == 429:
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = 60
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                wait_time = int(retry_after) + 5
                            except:
                                pass
                        print(f"   ⏰ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    return None
                
                else:
                    print(f"❌ DeepSeek API error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                    
            except Exception as e:
                print(f"Error querying DeepSeek: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
        
        return None
    
    def query_anthropic(self, 
                        model: str,
                        user_prompt: str,
                        system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                        temperature: float = 0,
                        max_retries: int = 5) -> Dict[str, Any]:
        """Query Anthropic's Claude API"""
        if not self.anthropic_api_key:
            print("❌ Anthropic API key not provided!")
            return None
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": model,
            "max_tokens": 500,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["content"][0]["text"]
                    return {"message": {"content": text}}
                
                elif response.status_code == 429:
                    print(f"⚠️  Rate limit (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        retry_after = response.headers.get('retry-after', 60)
                        time.sleep(int(retry_after))
                        continue
                    return None
                
                else:
                    print(f"❌ Error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
        
        return None
    
    def query_gemini(self, 
                     model: str,
                     user_prompt: str,
                     system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                     temperature: float = 0,
                     max_retries: int = 10) -> Dict[str, Any]:
        """Query Google Gemini API"""
        if not self.gemini_api_key:
            print("❌ Gemini API key not provided!")
            return None
        
        if not model.startswith("models/"):
            model = f"models/{model}"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={self.gemini_api_key}"
        
        full_prompt = f"{system_prompt}\n\nQuestion: {user_prompt}"
        
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 20000,
                "candidateCount": 1
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return {"message": {"content": text}}
                
                elif response.status_code == 429:
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = 60
                        print(f"   ⏰ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    return None
                
                else:
                    print(f"❌ Gemini API error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                    
            except Exception as e:
                print(f"Error querying Gemini: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
        
        return None
    
    def query_ollama(self, 
                     model: str, 
                     user_prompt: str, 
                     system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                     temperature: float = 0,
                     max_retries: int = 3) -> Dict[str, Any]:
        """Query Ollama model"""
        url = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 2048,
                "num_predict": 1024
            }
        }
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.unload_all_models()
                    time.sleep(10)
                    print(f"   Retrying after memory clear...")
                
                response = requests.post(url, json=payload, timeout=180)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 500 and "memory" in response.text.lower():
                    print(f"⚠️ Memory error on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        self.unload_model(model)
                        time.sleep(5)
                        continue
                else:
                    print(f"Error: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error querying model: {e}")
                if attempt < max_retries - 1:
                    continue
                return None
        
        return None
    
    def format_output(self, 
                      user_prompt: str,
                      system_prompt: str,
                      model: str,
                      temperature: float,
                      api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Format the response in the required structure"""
        return {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature,
            "responses": [
                {
                    "content": api_response.get('message', {}).get('content', ''),
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": []
                }
            ]
        }
    
    def save_response(self, 
                      pattern_name: str,
                      model_name: str,
                      example_number: int,
                      temperature: float,
                      response_data: Dict[str, Any],
                      method: str = "zero-shot"):
        """Save response to the appropriate JSON file"""
        output_dir = self.repo_path / "data" / method / model_name / pattern_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_str = str(temperature).replace('.', '_')
        filename = f"{model_name}_{pattern_name}_{example_number}_{temp_str}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Saved: {output_path.relative_to(self.repo_path)}")
    
    def process_file(self, 
                     json_file_path: str,
                     models: Optional[List[str]] = None,
                     temperature: float = 0,
                     system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                     method: str = "zero-shot",
                     use_gemini: bool = False,
                     use_deepseek: bool = False,
                     use_anthropic: bool = False,
                     use_openai: bool = False):
        """Process all prompts in a JSON file through specified models"""
        prompts = self.load_prompts(json_file_path)
        if not prompts:
            print(f"No prompts found in {json_file_path}")
            return
        
        pattern_name = prompts[0][0]
        
        if models is None:
            if use_openai:
                models = ["gpt-5"]
            elif use_anthropic:
                models = ["claude-haiku-4-5-20251001"]
            elif use_deepseek:
                models = ["deepseek-reasoner"]
            elif use_gemini:
                models = ["gemini-2.0-flash"]
            else:
                models = self.get_available_models()
                if not models:
                    print("No models available! Make sure Ollama is running.")
                    return
        
        total = len(prompts) * len(models)
        current = 0
        success_count = 0
        fail_count = 0
        
        print(f"\n{'='*70}")
        print(f"Processing: {pattern_name} ({len(prompts)} prompts)")
        print(f"Method: {method}")
        print(f"Models: {len(models)}")
        print(f"Temperature: {temperature}")
        print(f"{'='*70}\n")
        
        for model in models:
            model_dir_name = model.replace(':', '_').replace('/', '_').replace('-', '_')
            
            print(f"\n{'─'*70}")
            print(f"🤖 Model: {model}")
            print(f"{'─'*70}")
            
            for pattern, example_num, prompt_text in prompts:
                current += 1
                
                # Check if response already exists
                output_dir = self.repo_path / "data" / method / model_dir_name / pattern
                temp_str = str(temperature).replace('.', '_')
                filename = f"{model_dir_name}_{pattern}_{example_num}_{temp_str}.json"
                output_path = output_dir / filename
                
                if output_path.exists():
                    print(f"\n[{current}/{total}] Example {example_num}/{len(prompts)}")
                    print(f"⏭️  Skipping - already exists: {filename}")
                    success_count += 1
                    continue
                
                print(f"\n[{current}/{total}] Example {example_num}/{len(prompts)}")
                print(f"📝 Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
                
                # Query the appropriate API
                if use_openai:
                    response = self.query_openai(model, prompt_text, system_prompt, temperature)
                elif use_anthropic:
                    response = self.query_anthropic(model, prompt_text, system_prompt, temperature)
                elif use_deepseek:
                    response = self.query_deepseek(model, prompt_text, system_prompt, temperature)
                elif use_gemini:
                    response = self.query_gemini(model, prompt_text, system_prompt, temperature)
                else:
                    response = self.query_ollama(model, prompt_text, system_prompt, temperature)
                
                if response:
                    formatted_response = self.format_output(
                        user_prompt=prompt_text,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        api_response=response
                    )
                    
                    self.save_response(
                        pattern_name=pattern,
                        model_name=model_dir_name,
                        example_number=example_num,
                        temperature=temperature,
                        response_data=formatted_response,
                        method=method
                    )
                    
                    answer = formatted_response['responses'][0]['content'].strip()
                    print(f"💬 Response: {answer}")
                    success_count += 1
                else:
                    print(f"❌ Failed to get response for example {example_num}")
                    fail_count += 1
                
                time.sleep(0.5 if (use_gemini or use_deepseek or use_openai) else 1)
        
        print(f"\n{'='*70}")
        print(f"✅ Completed!")
        print(f"   - Successfully processed: {success_count}/{total}")
        if fail_count > 0:
            print(f"   - Failed: {fail_count}/{total}")
        print(f"{'='*70}\n")
    
    def process_multiple_files(self,
                               json_files: List[str],
                               models: Optional[List[str]] = None,
                               temperature: float = 0,
                               system_prompt: str = "Answer only with 'yes' or 'no' and nothing else.",
                               method: str = "zero-shot",
                               use_gemini: bool = False,
                               use_deepseek: bool = False,
                               use_anthropic: bool = False,
                               use_openai: bool = False):
        """Process multiple JSON files"""
        print(f"\n{'#'*70}")
        print(f"# BATCH PROCESSING: {len(json_files)} files")
        print(f"{'#'*70}")
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n\n📁 File {i}/{len(json_files)}: {Path(json_file).name}")
            self.process_file(
                json_file_path=json_file,
                models=models,
                temperature=temperature,
                system_prompt=system_prompt,
                method=method,
                use_gemini=use_gemini,
                use_deepseek=use_deepseek,
                use_anthropic=use_anthropic,
                use_openai=use_openai
            )
        
        print(f"\n{'#'*70}")
        print(f"# ALL FILES PROCESSED")
        print(f"{'#'*70}\n")


# ========================================================================
# LogiCue Pattern-Specific Prompts
# ========================================================================
LOGICUE_PROMPTS = {
    "CMP": """Critical reasoning about nested conditionals with probabilities:

Given three options X, Y, Z where P(X) > P(Y) > P(Z), if X doesn't occur, then Y becomes the most likely, NOT Z.

The nested conditional "if X doesn't happen, then if Y doesn't happen, Z will happen" does NOT make "if X doesn't happen, Z will happen" likely. Why? Because when X is eliminated, Y still has higher probability than Z.

Example: If P(X)=0.50, P(Y)=0.35, P(Z)=0.15, then P(Z|not X) = 0.15/(0.35+0.15) = 0.30, which is not likely.

The inference only holds when Z's probability is close to Y's, which contradicts the premise that Z has "much lower odds."

Answer with only 'yes' or 'no' - nothing else.""",

    "CT": """In classical logic, (P → Q) is equivalent to (¬Q → ¬P). This is valid for the material conditional. However, natural language "if" is not a material conditional. It implies a connection or dependence between P and Q.

Therefore, the inference from "If P, then Q" to "If not Q, then not P" is INVALID for natural language.

Counterexample: "If it's raining, then it's not raining hard" can be true (light rain). But its contrapositive, "If it's raining hard, then it's not raining," is a contradiction and is always false.

Now apply this reasoning to the following question and answer with only 'yes' or 'no' - nothing else.""",

    "ASd": """When evaluating conditional statements, remember that real-world conditionals often have implicit background assumptions and normal conditions built in. Don't apply formal logical rules mechanically without considering whether additional conditions in the antecedent might contradict these implicit assumptions.

After considering these factors, write 'Answer: ' followed by only 'yes' or 'no' and nothing else.""",

    "DSmi": """When evaluating logical inferences involving modal expressions like "might," "could," "possibly," etc.:

"Might not P" means "possibly not P" — it expresses uncertainty only

Neither "might not P" nor "not must P" logically entail "not P" — they only mean P is not necessary, but P could still be true

In natural language, do not treat "might not P" as equivalent to "not must P"

You cannot use disjunctive syllogism (Either A or B + Not B → A) with mere possibilities.

Example: From "Either A or B" + "A might not be true" you CANNOT conclude "B is true" because "might not" only indicates possibility, not definitive negation.

Only definitive statements like "not P" or "definitely not P" can be used in disjunctive syllogism.

After applying this reasoning, write 'Answer: ' followed by only 'yes' or 'no' and nothing else.""",

    "DSmu": """When evaluating arguments with "must," distinguish between:
- Content: "Fido is in the garden"
- Modal claim: "Fido must be in the garden" (necessity)

"Either A or B must be true" creates a disjunction between:
1. A
2. The necessity of B (not just B itself)

"It's not the case that B must be true" only negates the necessity - B could still be true, just not necessarily so.

Therefore: "Either A or B must be true" + "B need not be true" does NOT allow you to conclude A.

Example: "Either it's raining or it must be sunny" + "It need not be sunny" ≠ "It's raining"

After applying this reasoning, write 'Answer: ' followed by only 'yes' or 'no' and nothing else.""",

    "MTmi": """When working with logical inferences, carefully distinguish between definitive statements and statements about possibility/uncertainty. For modus tollens (If P then Q, not Q, therefore not P) to be valid, you need a definitive negation of Q, not just uncertainty about Q.

If given 'If P then Q' and 'Q might not be true' (or 'possibly not Q'), you can only conclude 'P might not be true' - NOT 'P is definitely false.'

Before concluding any logical inference, check: Do I have definitive premises or only statements about possibility? Match your conclusion's certainty level to your premises' certainty level.

Apply this reasoning to the following question and answer with only 'yes' or 'no' - nothing else.""",

    "MTmu": """Key rule: 'It's not the case that Q must be true' means Q is uncertain/optional, NOT that Q is definitely false.

Standard modus tollens: 'If P then Q' + 'not Q' → 'not P' is VALID
Modal modus tollens: 'If P then Q must be true' + 'Q need not be true' → 'P is false' is INVALID

Only definitive negation of Q allows concluding not-P.

Answer with only 'yes' or 'no' - nothing else."""
}


if __name__ == "__main__":
    # Initialize the runner with config
    runner = Runner(
        repo_path=PATHS.get('repo_path', '.'),
        ollama_url=PATHS.get('ollama_url', 'http://localhost:11434'),
        gemini_api_key=API_KEYS.get('gemini'),
        deepseek_api_key=API_KEYS.get('deepseek'),
        anthropic_api_key=API_KEYS.get('anthropic'),
        openai_api_key=API_KEYS.get('openai')
    )
    
    # Show available Ollama models
    print("\n" + "="*70)
    print("OLLAMA MODELS DETECTED:")
    print("="*70)
    models = runner.get_available_models()
    if models:
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    else:
        print("  No models found. Make sure Ollama is running!")
    print("="*70)
    
    # ========================================================================
    # OPTION 1: Process a single file with zero-shot
    # ========================================================================
    print("\n🚀 Starting processing...\n")
    
    runner.process_file(
        json_file_path="prompt/CMP.json",
        models=['gemma3:12b'],
        temperature=0,
        system_prompt="Answer only with 'yes' or 'no' and nothing else.",
        method="zero-shot"
    )
    
    # ========================================================================
    # OPTION 2: Process ALL patterns with zero-shot
    # ========================================================================
    #pattern_files = [
    #    "prompt/CMP.json",
    #    "prompt/ASd.json",
    #    "prompt/CT.json",
    #    "prompt/DSmi.json",
    #    "prompt/DSmu.json",
    #    "prompt/MTmi.json",
    #    "prompt/MTmu.json",
    #]
    
    #print("\n🚀 Starting batch processing of all patterns (zero-shot)...\n")
     
    #runner.process_multiple_files(
    #    json_files=pattern_files,
    #    models=['gpt-5'],
    #    temperature=0,
    #    system_prompt="Answer only with 'yes' or 'no' and nothing else.",
    #    method="zero-shot",
    #    use_openai=True
    #)
    
    # ========================================================================
    # OPTION 3: Process ALL patterns with LogiCue
    # ========================================================================
    #pattern_files = [
    #    "prompt/CMP.json",
    #    "prompt/ASd.json",
    #    "prompt/CT.json",
    #    "prompt/DSmi.json",
    #    "prompt/DSmu.json",
    #    "prompt/MTmi.json",
    #    "prompt/MTmu.json",
    #]
     
    #print("\n🚀 Starting LogiCue processing of all patterns...\n")
    #print("Each pattern will use its tailored LogiCue prompt for better reasoning.\n")
     
    #for pattern_file in pattern_files:
    #    pattern_name = Path(pattern_file).stem
    #    logicue_prompt = LOGICUE_PROMPTS.get(pattern_name, "Answer only with 'yes' or 'no' and nothing else.")
    #    print(f"\n{'='*70}")
    #    print(f"Processing {pattern_name} with tailored LogiCue prompt")
    #    print(f"{'='*70}")
    #    runner.process_file(
    #        json_file_path=pattern_file,
    #        models=['gpt-5'],
    #        temperature=0,
    #        system_prompt=logicue_prompt,
    #        method="LogiCue",
    #        use_openai=True,
    #    )
    
    # ========================================================================
    # OPTION 4: Process ALL patterns with Chain-of-Thought
    # ========================================================================
    #pattern_files = [
    #    "prompt/CMP.json",
    #    "prompt/ASd.json",
    #    "prompt/CT.json",
    #    "prompt/DSmi.json",
    #    "prompt/DSmu.json",
    #    "prompt/MTmi.json",
    #    "prompt/MTmu.json",
    #]

    #print("\n🚀 Starting Chain-of-Thought processing...\n")
    #print("Using step-by-step reasoning for all patterns.\n")

    #runner.process_multiple_files(
    #    json_files=pattern_files,
    #    models=['gpt-5'],
    #    temperature=0,
    #    system_prompt="In response to the following question, think step by step and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else. Please make sure to format your response by first explaining your reasoning and then writing 'Answer:' followed by 'yes' or 'no' at the very end",
    #    method="chain-of-thought",
    #    use_openai=True
    #)