# DeepAnalyze: Architecture & Compliance FAQ (Explained Simply!)

Got questions about how DeepAnalyze keeps your private company files safe while using cloud AI? Here are clear, friendly answers explained like you're in high school—no confusing legal or corporate jargon required!

---

### Q1: How does DeepAnalyze protect messy, unflattened spreadsheets with missing or weird columns?
* **The Analogy:** Imagine an old-school school notebook where someone glued sticky notes, receipts, and hand-written dates all over the page instead of writing in neat rows and columns.
* **The Problem:** Normal privacy scanners only look at column headers (like searching for a column named `Student Name`). If an invoice number or customer name is tucked away inside row 14 next to a random colon (`:`), normal scanners miss it completely!
* **The DeepAnalyze Fix:** DeepAnalyze uses **Cell-Level Geometric Masking**. Think of it like taking a photocopier, keeping the outline and form borders (`Doc. No`, `Doc Date`, `Seq`, `GL Code`, `:`), but putting black marker over all real client names (`XXXX`), invoice numbers (`XX-99999`), and money figures (`9,999.00`). The cloud AI can still see the puzzle's shape, but can't see any real secrets!

---

### Q2: What happens if I click "Not Sure" for the privacy law or dataset type?
* **The Analogy:** Like an automatic spell-checker or GPS that figures out what country you're driving in without you having to look at a paper map.
* **The DeepAnalyze Fix:** If you select **"Not Sure"** for the compliance rule, DeepAnalyze looks at your computer's country setting and picks the right law automatically (for example: Saudi Arabia $\rightarrow$ **Saudi PDPL & NDMO**; European Union $\rightarrow$ **GDPR**). If you select "Not Sure" for your dataset, it scans the spreadsheet for colons, ragged rows, and weird headers to figure out automatically if it's a messy accounting export or a normal table.

---

### Q3: How does the "value teaching" feature work?
* **The Analogy:** Teaching your phone to recognize your friend's unique nickname.
* **The DeepAnalyze Fix:** If the scanner misses an internal company code that only your company uses (like a special project code `500-000` or an employee badge `EMP-99`), you don't have to write any code. You just tell DeepAnalyze: *"Hey, look at column 3, here's an example: `500-000`."* DeepAnalyze instantly learns the pattern and automatically disguises all matching codes across thousands of rows!

---

### Q4: What is the difference between an "encrypted duplicate file" and a "clipboard payload"?
* **Encrypted Duplicate File (`[name]_anonymized.xlsx`):** Think of this as a **"stunt double"** of your entire Excel file. Every single row and sheet is preserved, but every name and dollar amount is replaced with safe fake placeholders. You can safely email this file or upload it directly to ChatGPT or Claude.
* **Clipboard Payload:** Think of this as a **"quick sample snapshot"**. DeepAnalyze creates a 5-row synthetic mini-example with safe fake numbers and copies it to your clipboard. You just press `Cmd+V` (or `Ctrl+V`) into your chat with ChatGPT, ask for a Python script, and you're done in 10 seconds!

---

### Q5: What happens if the cloud AI writes code that has a bug or error?
* **The Analogy:** A friendly spell-checker that doesn't crash your computer when a word is misspelled.
* **The DeepAnalyze Fix:** When you paste code from ChatGPT or Claude into DeepAnalyze, it tests the code safely inside temporary memory (RAM). If the code has a typo or error, DeepAnalyze catches it without crashing your session. It shows you the exact error and asks: `"Would you like to paste the corrected code? [y/N]"`. You just copy the error back to ChatGPT, get the fix, paste it, and keep going!

---

### Q6: How does DeepAnalyze guarantee zero data leaks while running code?
* **The Analogy:** An airport security scanner with a metal detector.
* **The DeepAnalyze Fix:** The cloud AI only ever sees fake stunt-double data. When the AI writes a Python cleaning script, our **AST Security Firewall (`firewall.py`)** scans the code before it runs. It strictly blocks any attempts to access the internet (`requests`, `urllib`, `socket`), steal environment variables, or delete files from your hard drive. The script runs strictly in your computer's RAM, puts the real names back into place, and leaves zero trace behind.

---

### Q7: What if the cloud AI gives me code using Pandas instead of Polars?
* **The Analogy:** A universal power adapter that works with both US and European electrical plugs.
* **The DeepAnalyze Fix:** DeepAnalyze has a **Dual-Engine Execution Layer**. Frontier AI models love writing code with `pandas` (`pd`) and `numpy` (`np`). DeepAnalyze automatically provides both `pandas` and `polars` in memory. If the incoming AI code uses Pandas (`df.iloc`, `df['col']`), DeepAnalyze handles it smoothly; if it uses Polars (`pl.col`), it handles that too. Everything works seamlessly either way!

---

### Q8: What if I don't know Python and just want to clean my data in Microsoft Excel?
* **The Analogy:** Having two checkout lanes at the grocery store: an express lane for coders, and a friendly full-service lane for Excel lovers.
* **The DeepAnalyze Fix:** DeepAnalyze gives you **Dual-Track Delivery**:
  * **Track A (Automated Python in RAM):** 1-click execution in memory that automatically exports `Clean_file.xlsx`.
  * **Track B (Power Query for Excel):** DeepAnalyze generates a ready-to-paste **Power Query M-Script** (`powerquery_script.m`) along with a step-by-step click guide (`powerquery_guide.md`). You can paste it into Excel's Advanced Editor, click apply, and clean your data without writing a single line of Python! Plus, you can refresh it on future monthly files with a single click.

---

### Q9: Why do some tools lose columns from messy spreadsheets, and how does DeepAnalyze keep them all?
* **The Analogy:** Reading only the first line of an envelope vs. opening the letter to see all the pages inside.
* **The DeepAnalyze Fix:** Naive tools only look at row 1. In messy ERP files, row 1 might only say `"Report Date: 2026"`, so dumb tools think the file only has 1 or 2 columns! DeepAnalyze uses a smart whole-sheet geometry scanner (`pd.read_excel(clean_path, header=None)`). It scans the entire sheet first, finds all 16+ columns, and makes sure not a single column or cell gets lost.

---

### Q10: How does DeepAnalyze stop people from guessing identities using clues like age and zip code?
* **The Analogy:** Blending into a crowd wearing matching school uniforms.
* **The DeepAnalyze Fix:** Removing names isn't enough if someone is the *only* 19-year-old living in a small village ($k=1$, easy to identify). DeepAnalyze uses **$k$-Anonymity** and **$l$-Diversity**:
  * **$k$-Anonymity:** Makes sure that every person in the dataset shares their general demographic traits (like age brackets `[20-29]`) with at least 4 other people ($k \ge 5$).
  * **$l$-Diversity:** Ensures that within any group of 5 matching people, their sensitive records (like diagnoses or salary levels) have at least 2 different values ($l \ge 2$), so nobody can deduce a private secret by guessing.

---

### Q11: How do the Real-Time Quality Scorecard and automated Pytest generator work?
* **The Analogy:** A teacher grading your paper and handing you back an automatic answer key to double-check future assignments.
* **The DeepAnalyze Fix:** At the end of the cleaning wizard, DeepAnalyze displays a **Quality Scorecard (0 to 100)** showing:
  * How many duplicate rows were removed.
  * What percentage of missing/null values were fixed.
  * Whether column names were cleaned into neat `snake_case`.
  
  At the exact same time, it creates a runnable test file (`test_clean_pipeline.py`). You can run `pytest` anytime to mathematically prove that your cleaned data meets 100% of your quality rules!
