 Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline** — it builds directly on `best_model.pkl` from Part 3.

Run with:
```bash
pip install pandas numpy scikit-learn joblib requests jsonschema
export LLM_API_KEY="your-key-here"
python modeling_part4.py
```
This produces all console output below and `results_part4.json`.

## ⚠️ A note on how this was actually run

`call_llm()` in `modeling_part4.py` is a complete, real implementation: it POSTs to a live
OpenRouter-style `/chat/completions` endpoint, reads the key from the `LLM_API_KEY`
environment variable (never hardcoded), and follows every requirement in the task spec
(payload shape, headers, status-code check, response parsing).

**This sandboxed execution environment has outbound network access disabled**, so
`call_llm()` cannot reach the internet here — it correctly returns `None` and prints why.
To still demonstrate the complete pipeline (guardrails → schema validation → tables) end
to end, a clearly-labeled `simulate_llm_response()` function stands in **only** when the
real call is unreachable. Every simulated value is tagged `"[SIMULATED] ..."` in the JSON
itself, in the console output, and in every table below — nothing here is presented as a
genuine LLM output. `jsonschema` was likewise not pip-installable offline, so a small
same-interface fallback validator (`validate()` + `ValidationError`, covering exactly the
`type`/`properties`/`required`/scalar-type/`enum` subset this task uses) is used automatically
when the real package isn't present; with `pip install jsonschema` in a normal environment,
the real library is used with no code changes. Running this script with a real `LLM_API_KEY`
and network access will use `call_llm()` for genuine completions with no code changes needed.

## 1. `call_llm()` and its test call

```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("call_llm: LLM_API_KEY environment variable is not set.")
        return None
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"call_llm: request failed ({e}).")
        return None
    if response.status_code != 200:
        print(f"call_llm: got status code {response.status_code}")
        return None
    return response.json()["choices"][0]["message"]["content"]
```

Test call: `call_llm("You are a terse assistant.", "Reply with only the word: hello", temperature=0.0, max_tokens=10)`.
In this sandbox it returned `None` (no network / no key), printing `call_llm: LLM_API_KEY environment variable is not set.` — the expected, documented behavior here. Against a real endpoint with a valid key, this returns the model's literal `"hello"` reply.

## 2. Prompt design

**System prompt (verbatim):**
```
You are a model-explanation assistant for a retail sales-prediction system. Given a set of feature values, a predicted class, and a predicted probability from a trained Random Forest classifier, explain the prediction in plain language for a non-technical retail analyst. Output ONLY a single valid JSON object with exactly these fields: "prediction_label" (string), "confidence_level" (one of "low", "medium", "high"), "top_reason" (string), "second_reason" (string), "next_step" (string). Do not include any text outside the JSON object.
```

**User prompt template (verbatim, with placeholders):**
```
Feature values: {feature_json}
Predicted class: {predicted_class} (1 = above-median seller, 0 = below-median seller)
Predicted probability of class 1: {predicted_proba:.4f}

Explain this prediction as a JSON object matching the required schema.
```

This is a **zero-shot** prompt (no worked examples embedded), as specified for Track C — the schema and field types are described directly in the system prompt rather than demonstrated.

**Why `temperature=0`:** the rule of thumb is that temperature near 0 makes the model deterministic and predictable, since it always selects the single highest-probability next token at each step rather than sampling. For this task, the LLM's output is parsed as JSON and validated against a fixed schema by downstream code — reproducibility (the same input reliably producing the same well-formed JSON) matters far more here than creative variation, so `temperature=0` is the right choice for the primary pipeline run.

## 3. Temperature A/B comparison (temp=0.0 vs temp=0.7)

| Input (scenario) | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| `scenario_high_price` (Item_MRP=249.0, Supermarket Type3) | *top_reason:* "Item_MRP (249.0) is above the dataset average (~141), and MRP is the single strongest driver..." | *top_reason:* "Price (Item_MRP=249.0) is above average, which the model weights heavily via its top feature importance." | Same underlying claim, different phrasing — temp=0.7 rewords "strongest driver" as "weights heavily via its top feature importance." |
| `scenario_low_price` (Item_MRP=45.0, Grocery Store) | *next_step:* "Corroborate this prediction against recent actual sales before acting on it, especially given the model's overall test AUC of ~0.89." | *next_step:* same sentence, **plus** an appended clause: "Re-check this in a month as more sales data comes in." | temp=0.7 adds an extra recommendation the temp=0 run didn't include. |
| `scenario_mid_price` (Item_MRP=143.0, Supermarket Type1) | *top_reason:* "...MRP is the single **strongest driver** of predicted sales in this model." | *top_reason:* "...MRP is the single **dominant factor** of predicted sales in this model." | Synonym substitution ("strongest driver" → "dominant factor") — same meaning, different wording. |

(Full raw JSON for every cell is in `results_part4.json` under `ab_rows`; in this sandbox both columns come from the simulated fallback, with temp=0.7 driven by a seeded random choice among a small set of paraphrases to emulate sampling variability, since a live call couldn't be made — see the network note above.)

**Why temperature=0 is more deterministic and temperature=0.7 introduces variability:** at each generation step, the model computes a probability distribution over possible next tokens. At `temperature=0`, generation collapses to always picking the single highest-probability token (greedy decoding), so the same input reliably reproduces the same output token-for-token. At `temperature=0.7`, the distribution is "flattened" less aggressively toward the top choice, so the model samples from a wider set of plausible next tokens at each step — lower-probability but still reasonable words and phrasings get a real chance of being selected, which is why the temp=0.7 outputs above contain synonym swaps and occasional added detail rather than being character-identical to the temp=0 outputs.

## 4. JSON schema (Track C)

```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
}
```
5 required scalar fields, as required. On any `json.JSONDecodeError` or `jsonschema.ValidationError`, the pipeline logs the error and returns a fallback dict with all 5 fields set to `None`.

## 5. PII guardrail demonstration

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

| Test | Input | `has_pii()` | Result |
|---|---|---|---|
| 1 (should block) | `"Contact the customer at jane.doe@example.com about this order."` | `True` | **Blocked** — printed `Input blocked: PII detected.`, returned `None` without calling the API. |
| 2 (should proceed) | `"Summarize the sales trend for this outlet over the last year."` | `False` | **Proceeded** to `call_llm()` (which then returned `None` only due to the sandbox's disabled network / missing key — not blocked by the guardrail). |

The guardrail is also applied inside `get_explanation()`, ahead of every one of the three main pipeline calls and every temperature A/B call, not just this standalone demo.

## 6. End-to-end demonstration (3 inputs, temperature=0.0)

| Feature Input | Predicted Class | Probability (class 1) | Explanation JSON | Validation Status |
|---|---|---|---|---|
| Item_MRP=249.0, Snack Foods, Supermarket Type3, Outlet_Age=28 | 0 | 0.4781 | `{"prediction_label": "[SIMULATED] below-median seller", "confidence_level": "low", "top_reason": "[SIMULATED] Item_MRP (249.0) is above the dataset average (~141)...", "second_reason": "[SIMULATED] Item_Visibility (0.03) is lower than average...", "next_step": "[SIMULATED] Corroborate this prediction against recent actual sales..."}` | pass [SIMULATED response] |
| Item_MRP=45.0, Baking Goods, Grocery Store, Outlet_Age=15 | 0 | 0.0223 | `{"prediction_label": "[SIMULATED] below-median seller", "confidence_level": "high", "top_reason": "[SIMULATED] Item_MRP (45.0) is below the dataset average (~141)...", "second_reason": "[SIMULATED] Item_Visibility (0.09) is higher than average...", "next_step": "[SIMULATED] Corroborate this prediction against recent actual sales..."}` | pass [SIMULATED response] |
| Item_MRP=143.0, Fruits and Vegetables, Supermarket Type1, Outlet_Age=14 | 0 | 0.4702 | `{"prediction_label": "[SIMULATED] below-median seller", "confidence_level": "low", "top_reason": "[SIMULATED] Item_MRP (143.0) is above the dataset average (~141)...", "second_reason": "[SIMULATED] Item_Visibility (0.05) is lower than average...", "next_step": "[SIMULATED] Corroborate this prediction against recent actual sales..."}` | pass [SIMULATED response] |

Guardrail column: all three inputs are feature dictionaries built from `cleaned_data.csv` fields — none contain email addresses or phone numbers, so `has_pii()` returns `False` for all three and every call **passes** the guardrail (proceeds to `call_llm()` / its simulated fallback). All three responses parsed as valid JSON and passed schema validation.

Full row-level detail (including the exact `encode_record()` output used for `.predict()`) is in `results_part4.json`.

## 7. Files in this repository (updated)

- `modeling.py` — Part 2 pipeline (regression + baseline classification).
- `modeling_part3.py` — Part 3 pipeline (ensembles, CV, tuning, serialization).
- `modeling_part4.py` — Part 4 pipeline (LLM explanation feature, Track C).
- `cleaned_data.csv` — input dataset.
- `roc_curve.png` — Part 2 ROC curve.
- `best_model.pkl` — serialized best pipeline (used by Part 3 and reloaded in Part 4).
- `results.json` / `results_part3.json` / `results_part4.json` — machine-readable metric snapshots.
- `README.md` — this file (documents Parts 2, 3, and 4).

