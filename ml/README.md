# Prompt-Injection Dataset for deBERTa-v3-large

## What I did
1. **Applied your 9 rule files (105 regex rules)** to the original 546-row dataset. Fixed 3 rules that had invalid Python regex syntax (`ENC-010`, `JBK-005`, `ROL-005` — they used non-Python `\x{...}` / mid-pattern `(?i)` syntax).
2. **Key finding:** the regexes only catch **10 of 203 (5%)** of the real human-labeled injection examples. Almost all real attacks in your data are semantic/instructional ("forget everything and do X", "you are now Xi Jinping...") rather than the marker-phrase style these rules target (`[SYSTEM]`, "ignore previous instructions", etc.). Two benign rows also false-triggered `ENC-010` (stray zero-width characters from copy-pasted German text). This is exactly why a trained classifier is more useful than regex alone — I kept this in mind while building the set below.
3. **Generated synthetic positive examples** (308+6=314) that each verifiably match one specific rule, using `exrex` to sample real regex-matching fragments and wrapping them in natural sentence templates — one set per rule category, covering **105/105 rules**.
4. **Generated hard-negative benign examples** (25) — legitimate requests using similar vocabulary (e.g. "send this report to my manager", "explain ROT13 for a school project"). Several of these *did* trip your regexes (`EXF-006`, `ENC-003`, `ENC-007`, `JBK-011`, `ROL-018`), confirming those rules are prone to false positives on ordinary phrasing — I kept them labeled `0` (benign) on purpose so the model learns to be more discriminating than the regex.
5. Merged everything, deduplicated, and split **80/10/10** stratified by label.

## Final dataset
`prompt_injection_full_labeled.{csv,parquet}` — 885 rows total:
- `text`, `label` (0=benign, 1=injection)
- `category` — 9 attack categories from your rules + `benign` + `semantic_injection` (real attacks no rule caught)
- `matched_rule_ids`, `source` (`original` / `synthetic_rule` / `synthetic_hard_negative`)

| split | rows | label=1 | label=0 |
|---|---|---|---|
| train | 708 | 414 | 294 |
| validation | 88 | 51 | 37 |
| test | 89 | 52 | 37 |

Ready-to-train files (what `train_deberta.py` expects): `train.csv`, `validation.csv`, `test.csv` (just `text,label`).
Also included `*_with_category.csv` + `category_label_map.json` if you want a secondary multi-class head for attack-type classification instead of/in addition to binary.

`rules_compiled.json` — your 105 rules flattened with the 3 regex fixes applied, in case you want to keep using them alongside the model (e.g. regex as a fast pre-filter, deBERTa for the semantic cases it misses).

## Training
```bash
pip install transformers datasets accelerate scikit-learn torch
python train_deberta.py
```
Binary classification, 5 epochs, lr 1e-5, fp16 — adjust batch size/epochs for your GPU. The script reports accuracy/precision/recall/F1 on the held-out test split.

## Caveats to keep in mind
- 885 rows is still small for `deberta-v3-large` (304M params) — expect it to overfit fast; watch validation loss, consider fewer epochs, LoRA, or gathering more real examples (the `deepset/prompt-injections` dataset your 546 rows came from, or a public jailbreak dataset, would extend this well).
- The synthetic examples are regex-shaped, so the model may lean on the same surface cues the rules do. Weight the `semantic_injection` and `original` rows highly during evaluation — that's your real signal for whether the model generalizes past the regexes.
