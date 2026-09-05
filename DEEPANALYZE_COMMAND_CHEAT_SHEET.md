# DeepAnalyze: Command Reference & Cloud AI Integration

A concise operational reference for DeepAnalyze v4.0 Zero-Code Air-Gap Gateway.

---

## Interactive Wizard Directives

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `%deepanalyze` | Full Zero-Code Interactive Wizard | `%deepanalyze` |
| `--airgap` | Direct Anonymization to Clipboard | `%deepanalyze --airgap --origin "Saudi Arabia" --jurisdiction "PDPL" --target df "Clean dates"` |
| `--run` | Audit and Execute External Code | `%%deepanalyze --run --target df` |
| `--undo` | Instant State Rollback (5 levels) | `%deepanalyze --undo --target df` |
| `--audit` | Export Compliance Certificate | `%deepanalyze --audit --out compliance_audit.md` |

---

## 13-Step Interactive Wizard Flow (`%deepanalyze`)

1. **Step 1:** Enter file path (drag-and-drop quotes automatically sanitized) or variable name. Preserves all 16+ columns.
2. **Step 2:** Operating country (Origin).
3. **Step 3:** Target statutory framework (Select country options or **"Not Sure"** for auto-detection).
4. **Step 4:** Dataset architecture (Select ERP, Tabular, EHR, or **"Not Sure"** for auto-detection).
5. **Step 5:** Deep scan & pattern categorization across all rows and cells (including free-text clinical notes).
6. **Step 6:** Review unique masked pattern snippet table and k-Anonymity / l-Diversity re-identification risk audit.
7. **Step 7:** Value teaching loop (*"Are there more columns to encrypt?"* -> enter example value like `500-000` to teach pattern).
8. **Step 8:** Export encrypted duplicate file (`[file]_anonymized.xlsx`) or copy 5-row Differential Privacy synthetic mock to clipboard.
9. **Step 9:** Code provision prompt (*"Will code be provided?"* -> select Single Script `.py`, Multiple Blocks `.ipynb`, or Power Query `.m`).
10. **Step 10:** Code paste, syntax preview, Enter to execute, and AST Firewall path/timing sandbox verification.
11. **Step 11:** Error self-healing loop (catches errors and allows pasting corrected code without crashing session).
12. **Step 12:** Real-Time Quality Scorecard, automatic detokenization, clean export (`Clean_file.xlsx`), and automated Pytest generation (`test_clean_pipeline.py`).
13. **Step 13:** Automatic generation of verifiable `compliance_audit.md` certificate.
