"""Run the required SQL queries and compare SQL JOIN output with pandas.merge."""
from pathlib import Path
import sqlite3
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE / "zepto_books.db"
OUT = HERE / "query_outputs.txt"

QUERIES = {
    "1_select_where": "SELECT title, price_gbp FROM books WHERE price_gbp > 30 ORDER BY price_gbp DESC;",
    "2_order_by_limit": "SELECT title, rating, price_gbp FROM books ORDER BY rating DESC, price_gbp DESC LIMIT 10;",
    "3_distinct": "SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
    "4_between": "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 1000 AND 2500 ORDER BY price_inr;",
    "5_join": """
        SELECT b.title, b.rating, b.price_inr, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title
        LIMIT 10;
    """,
}


def main() -> None:
    if not DB.exists():
        raise FileNotFoundError("Run scrape_and_load.py first.")
    with sqlite3.connect(DB) as conn:
        outputs = []
        for name, sql in QUERIES.items():
            frame = pd.read_sql_query(sql, conn)
            outputs.append(f"\n### {name}\nSQL: {sql.strip()}\n\n{frame.to_string(index=False)}\n")

        join_sql = QUERIES["5_join"]
        sql_join = pd.read_sql_query(join_sql, conn)
        books = pd.read_sql_query("SELECT * FROM books", conn)
        categories = pd.read_sql_query("SELECT * FROM categories", conn)
        merged = books.merge(categories, on="category_id", how="inner")
        merged = merged[["title", "rating", "price_inr", "category_name"]].sort_values(
            ["rating", "category_name", "title"], ascending=[False, True, True]
        ).head(10).reset_index(drop=True)
        sql_join = sql_join.reset_index(drop=True)
        equivalent = sql_join.equals(merged)
        outputs.append("\n### pd.read_sql vs pd.merge\n")
        outputs.append("SQL JOIN result:\n" + sql_join.to_string(index=False))
        outputs.append("\npd.merge result:\n" + merged.to_string(index=False))
        outputs.append(f"\nEquivalent: {equivalent}\n")

    OUT.write_text("\n".join(outputs), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
