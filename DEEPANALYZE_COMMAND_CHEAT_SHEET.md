# DeepAnalyze: Quick Command Cheat Sheet (High-School Friendly Edition!)

Need a quick cheat sheet on how to use DeepAnalyze to clean messy data with Cloud AI safely? Here's the simple, zero-jargon breakdown!

---

## The 5 Main Commands

| Magic Command | What It Does in Plain English | Real-Life Analogy | Example |
| :--- | :--- | :--- | :--- |
| `%deepanalyze` | **The Full Interactive Wizard:** Walks you step-by-step through inspecting, disguising, cleaning, and exporting your spreadsheet. | Like a friendly video game setup wizard guiding you through each level. | `%deepanalyze` |
| `--airgap` | **Fast Anonymization to Clipboard:** Takes your data, strips away all personal info, and copies a safe sample right to your clipboard. | Like a photocopier with an automatic black marker over sensitive names. | `%deepanalyze --airgap --origin "Saudi Arabia" --jurisdiction "PDPL" --target df "Clean dates"` |
| `--run` | **Safe Code Airlock:** Takes Python code from ChatGPT/Claude, checks it with a security scanner, and runs it safely in memory. | Like an airport luggage scanner that checks for contraband before letting code run. | `%%deepanalyze --run --target df` |
| `--undo` | **Time Machine Undo:** If an AI script messes up your dataset, this jumps back to how your data was before (up to 5 steps back!). | Like pressing `Ctrl+Z` in a video game or document editor. | `%deepanalyze --undo --target df` |
| `--audit` | **Generate Report Card:** Creates a verifiable compliance certificate proving no private data leaked over the internet. | Like an official school diploma or inspection stamp. | `%deepanalyze --audit --out compliance_audit.md` |

---

## The 13-Step Interactive Wizard Flow (`%deepanalyze`)

When you run `%deepanalyze`, here is what happens step by step:

1. **Step 1: Pick Your File:** Enter your file path (you can just drag-and-drop the file from your desktop!). DeepAnalyze reads all 16+ columns, even if row 1 is messy.
2. **Step 2: Tell Us Your Country:** DeepAnalyze asks where your data is from so it knows what privacy laws apply.
3. **Step 3: Choose Privacy Law:** Pick your statutory rule (like Saudi **PDPL**, European **GDPR**, or US **HIPAA**). If you aren't sure, just choose **"Not Sure"** and DeepAnalyze figures it out automatically!
4. **Step 4: Choose Dataset Style:** Is it an accounting export, medical chart, or normal table? Pick **"Not Sure"** to let DeepAnalyze auto-detect it.
5. **Step 5: The Deep Scan:** DeepAnalyze searches through every row, column, and note to find names, phone numbers, addresses, and ID cards.
6. **Step 6: The Privacy Inspection:** DeepAnalyze shows you a preview table of what it disguised and verifies that people can't be singled out ($k$-Anonymity $k \ge 5$).
7. **Step 7: Teach DeepAnalyze Custom Codes:** Did it miss a special company code like `500-000`? Just type one example and DeepAnalyze learns the pattern across the whole file!
8. **Step 8: Grab Your Safe Disguised Data:** 
   * Option A: Download a complete "stunt double" Excel file (`[file]_anonymized.xlsx`).
   * Option B: Copy a 5-row synthetic mini-mock directly to your clipboard to paste into ChatGPT or Claude.
9. **Step 9: Choose Your Code Style:** Tell DeepAnalyze what kind of code you want to run: Single Python script (`.py`), notebook blocks (`.ipynb`), or Microsoft Excel Power Query (`.m`).
10. **Step 10: Paste & Safety Check:** Paste the code you got from the AI. The AST Security Firewall inspects every line to make sure it doesn't try to touch the internet or delete files.
11. **Step 11: Error Self-Healing:** Did the AI write code with a typo? DeepAnalyze catches the error safely, lets you copy the error message back to the AI, and lets you paste the fix without crashing!
12. **Step 12: Grade Card & Clean Export:** You get a 0–100 Quality Score, real names are automatically restored from the private in-memory vault, your clean file is saved (`Clean_file.xlsx`), and a test file (`test_clean_pipeline.py`) is written for you.
13. **Step 13: Official Certificate:** DeepAnalyze writes an official certificate (`compliance_audit.md`) proving your data handling was 100% legal and private.
