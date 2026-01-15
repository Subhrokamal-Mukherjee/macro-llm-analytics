# Macro LLM Analytics

A constrained, auditable macroeconomic analysis system that combines
deterministic economic data with a locally run LLM.



 ## What this project does

- Aggregates macroeconomic indicators over user-specified periods
- Generates factual, professional economic descriptions using a local LLM
- Enforces strict grounding rules to prevent:
&nbsp; - causal speculation
&nbsp; - policy intent inference
&nbsp; - historical narrative leakage
- Validates LLM output against deterministic constraints

Large Language Models are powerful but unreliable narrators.
This project treats the LLM as an untrusted component and enforces
explicit epistemic boundaries through validation.

## Tech stack



- Python
- pandas
- llama.cpp (local inference)
- Rule-based validation

## Design philosophy

This project treats the LLM as an unreliable narrator.

All economic facts are derived deterministically from data.
The language model is used only to verbalize those facts under strict rules.

Key principles:
- No causal explanations unless explicitly present in data
- No policy intent or market expectation inference
- No historical or narrative leakage
- Explicit validation of LLM output against numeric constraints

If the model violates these rules, the response is rejected.



## Example



```text

Ask a macroeconomic question:

> 2009 Economic Conditions








