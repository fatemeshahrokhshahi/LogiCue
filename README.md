# LogiCue: Targeted Prompting for Improved Modal and Conditional Reasoning in Large Language Models

[![Paper](https://img.shields.io/badge/Paper-ACLing%202025-blue)](https://www.sciencedirect.com/journal/procedia-computer-science)
[![Conference](https://img.shields.io/badge/Conference-ACLing%202025-green)](https://acling.org/)

**Authors:** Fatemeh Shahrokhshahi, Farzan Mohammadi 
**Affiliation:** Department of Computer Engineering, Istanbul Aydin University  
**Status:** Accepted at ACLing 2025 (7th International Conference on AI in Computational Linguistics)  
**Publication:** Awaiting publication in Procedia Computer Science (Elsevier)

---

## Overview

Large language models systematically fail on fundamental logical inferences involving conditionals and modal operators (might, must). LogiCue addresses these failures through pattern-specific prompting strategies that target the root causes of reasoning errors.

**Key Results:**
- **82.8%** average accuracy with LogiCue vs. **31.9%** zero-shot baseline
- Tested across 9 models from 3B to frontier-scale parameters
- Improvements of **+13% to +64%** on challenging inference patterns

---

## Installation

```bash
git clone https://github.com/fatemeshahrokhshahi/LogiCue.git
cd LogiCue
pip install -r requirements.txt
```

## Configuration

1. Copy the configuration template:
```bash
cp config.example.py config.py
```

2. Add your API keys to `config.py`:
```python
API_KEYS = {
    'openai': 'your-key-here',
    'anthropic': 'your-key-here',
    'deepseek': 'your-key-here',
    'gemini': 'your-key-here'
}
```

---

## Usage

### Running Experiments

Edit `runner.py` to select your experiment configuration, then run:

```bash
python runner.py
```

### Available Methods

The framework supports four prompting approaches:

1. **Zero-Shot**: Baseline performance without guidance
2. **Chain-of-Thought**: Step-by-step reasoning instructions
3. **LogiCue**: Pattern-specific prompts addressing root causes of errors
4. **Binary LogiCue**: Modal reasoning with binary assumption testing

See `PROMPTS.md` for detailed prompt specifications.

---

## Repository Structure

```
LogiCue/
├── runner.py                          # Main experiment runner
├── config.example.py                  # Configuration template
├── requirements.txt                   # Python dependencies
├── PROMPTS.md                         # LogiCue prompt documentation
│
├── prompt/                            # Inference pattern datasets
│   ├── MTmu.json                      # Modus Tollens (modal unnegation)
│   ├── MTmi.json                      # Modus Tollens (modal indefinite)
│   ├── DSmu.json                      # Disjunctive Syllogism (modal unnegation)
│   ├── DSmi.json                      # Disjunctive Syllogism (modal indefinite)
│   ├── CT.json                        # Contraposition
│   ├── ASd.json                       # Antecedent Strengthening (defeasible)
│   └── CMP.json                       # Complex Modus Ponens
│
├── data/                              # Experimental results
│   ├── zero-shot/                     # Zero-shot responses
│   ├── chain-of-thought/              # Chain-of-Thought responses
│   └── LogiCue/                       # LogiCue responses
│
├── Error_taxonomy/                    # Error analysis
│   ├── Error_analysis.py              # Error extraction script
│   ├── [Pattern]_errors_for_analysis.txt
│   └── Classification/                # Error classification results
│       ├── Results/
│       └── Visualization/
│
└── graphs/                            # Performance visualizations
    └── multi_method_analysis/
        ├── Results/                   # Statistical summaries
        └── Visualization/             # Performance charts
```

---

## Inference Patterns

The framework evaluates seven challenging inference patterns:

| Pattern | Description | Core Challenge |
|---------|-------------|----------------|
| **MTmu** | Modus Tollens with "must" | Confusing necessity denial with definitive negation |
| **MTmi** | Modus Tollens with "might" | Using uncertainty as definitive negation |
| **DSmu** | Disjunctive Syllogism with "must" | Confusing modal claims with propositional content |
| **DSmi** | Disjunctive Syllogism with "might" | Confusing possibility with definitive negation |
| **CT** | Contraposition | Material vs. natural language conditionals |
| **ASd** | Antecedent Strengthening | Defeating implicit normalcy assumptions |
| **CMP** | Complex Modus Ponens | Probability preservation in nested conditionals |

---

## Results

### Overall Performance

| Model | Zero-Shot | LogiCue | Improvement |
|-------|-----------|---------|-------------|
| GPT-5 | 19.3% | 58.6% | +39.3% |
| Claude Sonnet 4.5 | 11.4% | 62.9% | +51.5% |
| DeepSeek Reasoner | 36.4% | 97.9% | +61.5% |
| Gemini 2.0 Flash | 15.7% | 87.9% | +72.2% |
| Claude Haiku 4.5 | 48.6% | 84.3% | +35.7% |

See the paper for comprehensive results and statistical analysis.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{shahrokhshahi2025logicue,
  title={LogiCue: Targeted Prompting for Improved Modal and Conditional Reasoning in Large Language Models},
  author={Shahrokhshahi, Fatemeh and Mohammadi, Farzan and Sonmez, Ferdi},
  booktitle={Proceedings of the 7th International Conference on AI in Computational Linguistics (ACLing 2025)},
  year={2025},
  publisher={Elsevier},
  series={Procedia Computer Science},
  note={To appear}
}
```

---

## Dataset

The inference patterns are based on the dataset from:

> Holliday, W. H., Mandelkern, M., & Zhang, C. E. (2024). Conditional and Modal Reasoning in Large Language Models. In *Proceedings of EMNLP 2024* (pp. 3800-3821).

Original dataset: https://github.com/wesholliday/llm-logic

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

**Fatemeh Shahrokhshahi**  
Istanbul Aydin University  
Email: fatemehshahrokhshahi@stu.aydin.edu.tr

For questions about the methodology or results, please open an issue or contact the authors directly.

---

## Acknowledgments

This work was conducted at the Department of Computer Engineering, Istanbul Aydin University. We thank the reviewers at ACLing 2025 for their valuable feedback.