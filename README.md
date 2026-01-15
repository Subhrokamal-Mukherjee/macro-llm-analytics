\# Macro LLM Analytics



A constrained, auditable macroeconomic analysis system that combines

deterministic economic data with a locally run LLM.



\## What this project does



\- Aggregates macroeconomic indicators over user-specified periods

\- Generates factual, professional economic descriptions using a local LLM

\- Enforces strict grounding rules to prevent:

&nbsp; - causal speculation

&nbsp; - policy intent inference

&nbsp; - historical narrative leakage

\- Validates LLM output against deterministic constraints



Large Language Models are powerful but unreliable narrators.

This project treats the LLM as an untrusted component and enforces

explicit epistemic boundaries through validation.



\## Tech stack



\- Python

\- pandas

\- llama.cpp (local inference)

\- Rule-based validation



\## Example



```text

Ask a macroeconomic question:

> 2009 Economic Conditions



