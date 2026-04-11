# Personal trade journal (optional)

Put your **private** RKLB (or other) trade log here as CSV. Recommended filename: `my_trades.csv`.

- Copy column ideas from **`configs/examples/journal_trades.example.csv`**.
- Required for **`sparkles journal compare`**: a date column named **`entry_date`**, **`date`**, **`open_date`**, or **`entry`** (ISO `YYYY-MM-DD`).
- Optional **`symbol`** / **`ticker`**: rows are filtered to the experiment **`symbol`** when this column is present.
- **Long holds:** one row per **open** is enough; `exit_date` can be months later or blank. Comparison uses **entry session date** only—what the model saw on labeled 1m **entry** bars that day, not daily P&L over the hold.

`*.csv` in this folder is **gitignored** so your history stays local. The **example** template lives under **`configs/examples/`**.
