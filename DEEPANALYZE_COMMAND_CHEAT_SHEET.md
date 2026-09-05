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

1. **Step 1:** Enter file path (drag-and-drop quotes automatically sanitized) or variable name.
2. **Step 2:** Operating country (Origin).
3. **Step 3:** Target statutory framework (Select country options or **"Not Sure"** for auto-detection).
4. **Step 4:** Dataset architecture (Select ERP, Tabular, EHR, or **"Not Sure"** for auto-detection).
5. **Step 5:** Deep scan & pattern categorization across all rows and cells.
6. **Step 6:** Review unique masked pattern snippet table.
7. **Step 7:** Value teaching loop (*"Are there more columns to encrypt?"* -> enter example value like `500-000` to teach pattern).
8. **Step 8:** Export encrypted duplicate file (`[file]_anonymized.xlsx`) or copy 5-row synthetic mock payload to clipboard.
9. **Step 9:** Code provision prompt (*"Will code be provided?"* -> select Single Script `.py` or Multiple Blocks `.ipynb`; supports Pandas & NumPy natively).
10. **Step 10:** Code paste, syntax preview, Enter to execute, and AST Firewall safety verification.
11. **Step 11:** Error self-healing loop (catches errors and allows pasting corrected code without crashing session).
12. **Step 12:** Automatic detokenization & export of clean dataset (`Clean_file.xlsx`).
13. **Step 13:** Automatic generation of verifiable `compliance_audit.md` certificate.
