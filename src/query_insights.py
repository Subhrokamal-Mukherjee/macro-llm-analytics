import re
import subprocess
from pathlib import Path
import pandas as pd


# =================================================
# Paths & configuration
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLAMA_BIN = (
    PROJECT_ROOT
    / "external"
    / "llama.cpp"
    / "build"
    / "bin"
    / "Release"
    / "llama-cli.exe"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "external"
    / "llama.cpp"
    / "models"
    / "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "macro_features_with_regime.csv"
)


# =================================================
# LLM execution
# =================================================

def run_llama(prompt: str) -> str:
    """
    Run a local LLM via llama.cpp and return raw stdout.
    """
    process = subprocess.run(
        [
            str(LLAMA_BIN),
            "-m", str(MODEL_PATH),
            "-p", prompt,
            "-no-cnv",
            "-ngl", "40",
            "--temp", "0.2",
            "--n-predict", "300",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())

    return process.stdout.strip()


# =================================================
# Question parsing
# =================================================

def extract_years(question: str):
    """
    Extract years like 2008 or ranges like 2007–2009.
    """
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    years = sorted(set(map(int, years)))

    if len(years) >= 2:
        return years[0], years[-1]
    if len(years) == 1:
        return years[0], years[0]
    return None


# =================================================
# Data loading
# =================================================

df = pd.read_csv(
    FEATURES_PATH,
    parse_dates=["date"]
)


# =================================================
# Data filtering
# =================================================

def filter_by_question(df: pd.DataFrame, question: str) -> pd.DataFrame:
    years = extract_years(question)

    if not years:
        return df

    start, end = years
    return df.loc[
        (df["date"].dt.year >= start) &
        (df["date"].dt.year <= end)
    ]


# =================================================
# Deterministic summarization
# =================================================

def summarize_period(df_slice: pd.DataFrame) -> dict:
    if df_slice.empty:
        raise ValueError("No data available for the specified period.")

    avg_real = round(df_slice["real_policy_rate"].mean(), 2)
    avg_infl = round(df_slice["inflation_yoy"].mean(), 2)
    avg_spread = round(df_slice["yield_spread"].mean(), 2)

    return {
        "start_date": df_slice["date"].min().date().isoformat(),
        "end_date": df_slice["date"].max().date().isoformat(),
        "avg_real_policy_rate": avg_real,
        "avg_inflation": avg_infl,
        "avg_yield_spread": avg_spread,
        "real_rate_sign": "negative" if avg_real < 0 else "positive",
        "yield_curve_shape": (
            "inverted" if avg_spread < 0 else "upward sloping"
        ),
        "dominant_regime": int(df_slice["regime"].mode().iloc[0]),
    }


# =================================================
# Prompt construction
# =================================================

def build_prompt(question: str, summary: dict) -> str:
    return f"""
You are an economic analyst producing a factual interpretation.- Prefer using the provided qualitative descriptors verbatim.
Adhere to the following guidelines when describing the macroeconomic environment:

STRICT RULES:
- USE ONLY the information provided in the data summary.
- Do NOT infer causes or drivers.
- Do NOT introduce external events, sectors, or historical narratives.
- Ensure all statements are numerically consistent with the data.
- If the data is insufficient to confirm a label or interpretation, say so explicitly.
- Do NOT contradict the qualitative indicators provided (rate sign, curve shape).
- Do NOT infer policy intent, stance, or market expectations.
- When describing interest rates, use only numeric or directional language
  (e.g., "positive", "negative", "above inflation", "below inflation").
- Do NOT use policy stance or intent terms such as "accommodative" or "restrictive".
- USE the provided definitions verbatim; do not restate or redefine metrics.

Period: {summary['start_date']} to {summary['end_date']}

Data summary:
- Average real policy rate: {summary['avg_real_policy_rate']}%
- Average inflation (YoY): {summary['avg_inflation']}%
- Average yield spread (defined as 10-year yield minus nominal policy rate): {summary['avg_yield_spread']}%
- Dominant macro regime (clustered): {summary['dominant_regime']}
- Real policy rate sign: {summary['real_rate_sign']}
- Yield curve shape: {summary['yield_curve_shape']}
- Interest rate description: real policy rate is {summary['real_rate_sign']}
  (No semantic meaning is provided for regime labels.)

Task:
Describe the macroeconomic environment during this period in clear, professional language.
Avoid speculation, prediction, and external context.

Question:
{question}

Answer:
""".strip()


# =================================================
# Output extraction (critical fix)
# =================================================

def extract_answer_only(raw_text: str) -> str:
    """
    Strip prompt echo and return only the model's answer.
    """
    if "Answer:" in raw_text:
        return raw_text.split("Answer:", 1)[1].strip()
    return raw_text.strip()


# =================================================
# Validation layer
# =================================================
FORBIDDEN_PHRASES = [
    # causal explanations
    "driven by",
    "caused by",
    "due to",
    "result of",

    # policy intent / stance
    "accommodative",
    "restrictive",
    "tightening",
    "easing",
    "policy stance",

    # expectations / forward-looking
    "expects",
    "expectations",
    "anticipated",
    "indicative of",
    "signals",
    "suggests that rates will",
    "future interest rate",
]



def validate_response(text: str, summary: dict):
    issues = []
    lowered = text.lower()

    # Causal language
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            issues.append(f"Introduced causal explanation: '{phrase}'")

    # Yield curve consistency
    if summary["avg_yield_spread"] > 0:
        if "inverted" in lowered or "negative spread" in lowered:
            issues.append("Positive yield spread described as inverted/negative")

    if summary["avg_yield_spread"] < 0:
        if "positive" in lowered or "upward sloping" in lowered:
            issues.append("Negative yield spread described as positive")

    return issues


# =================================================
# End-to-end execution
# =================================================

def answer_question(question: str) -> str:
    df_slice = filter_by_question(df, question)
    summary = summarize_period(df_slice)

    prompt = build_prompt(question, summary)
    raw_output = run_llama(prompt)
    answer_text = extract_answer_only(raw_output)

    issues = validate_response(answer_text, summary)

    if issues:
        issue_text = "\n".join(f"- {i}" for i in issues)
        return (
            "The generated response violated grounding rules:\n\n"
            f"{issue_text}\n\n"
            "Validated answer text:\n\n"
            f"{answer_text}"
        )

    return answer_text


# =================================================
# CLI entry point
# =================================================

if __name__ == "__main__":
    question = input("Ask a macroeconomic question:\n> ").strip()
    print("\n--- Answer ---\n")
    print(answer_question(question))
    print("\n---------------\n")
