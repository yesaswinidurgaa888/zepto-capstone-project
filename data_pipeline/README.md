# Data Pipeline

## What this module does

1. Scrapes the first five catalogue pages of `books.toscrape.com` using `requests` and BeautifulSoup.
2. Cleans price, rating and availability fields.
3. Converts GBP to INR using the assignment's fixed rate: **1 GBP = 105.50 INR**.
4. Stores the result in normalized SQLite tables `categories` and `books`.
5. Runs five SQL queries and compares the JOIN result with `pandas.merge`.

## Run

From the repository root:

```bash
python data_pipeline/scrape_and_load.py
python data_pipeline/queries.py
```

The scraper needs internet access. The database is regenerated from scratch by `scrape_and_load.py`.

## Cleaning decisions

- `price_gbp`: currency symbol is removed and the value is parsed as float. Numeric parse failures are median-imputed.
- `rating`: English star words are mapped One–Five → 1–5. Unexpected numeric parse failures are median-imputed and rounded to an integer in the valid range.
- `in_stock`: the availability text is converted to a boolean based on whether it contains `In stock`.
- Rows are not silently discarded for ordinary numeric parse failures because the specification explicitly allows median imputation for numeric fields.
- `price_inr` is always `price_gbp * 105.50`; this is the required artificial project baseline, not a live exchange rate.
