"""Scrape books.toscrape.com, clean the data, and load a normalized SQLite DB."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from statistics import median

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50
OUT_DIR = Path(__file__).resolve().parent
DB_PATH = OUT_DIR / "zepto_books.db"


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def scrape_books(min_rows: int = 60) -> pd.DataFrame:
    """Scrape the first five catalogue pages, which provide 100 books."""
    rows: list[dict] = []
    for page in range(1, 6):
        url = BASE_URL if page == 1 else f"{BASE_URL}catalogue/page-{page}.html"
        soup = get_soup(url)
        for article in soup.select("article.product_pod"):
            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            rating_tag = article.select_one("p.star-rating")
            availability_tag = article.select_one(".availability")
            rows.append(
                {
                    "title": title_tag.get("title", "").strip() if title_tag else "",
                    "price": price_tag.get_text(" ", strip=True) if price_tag else "",
                    "star_rating": " ".join(rating_tag.get("class", [])[1:]) if rating_tag else "",
                    "availability": availability_tag.get_text(" ", strip=True) if availability_tag else "",
                    # The product page contains the authoritative category; follow it below.
                    "product_url": requests.compat.urljoin(url, title_tag.get("href", "")) if title_tag else "",
                }
            )

    # Resolve category from each product page. This keeps category extraction explicit.
    for row in rows:
        try:
            product_soup = get_soup(row["product_url"])
            crumbs = [x.get_text(" ", strip=True) for x in product_soup.select("ul.breadcrumb li")]
            row["category"] = crumbs[-2] if len(crumbs) >= 2 else "Unknown"
        except requests.RequestException:
            row["category"] = "Unknown"

    df = pd.DataFrame(rows).drop(columns=["product_url"])
    if len(df) < min_rows:
        raise RuntimeError(f"Scrape produced only {len(df)} rows; expected at least {min_rows}.")
    return df


def clean_books(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["price_gbp"] = pd.to_numeric(
    df["price"].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
    errors="coerce"
)

    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    df["rating"] = df["star_rating"].map(rating_map)
    df["in_stock"] = df["availability"].astype(str).str.contains("In stock", case=False, na=False)

    # Numeric parse failures use the required median-imputation approach.
    price_median = df["price_gbp"].median()
    rating_median = df["rating"].median()
    df["price_gbp"] = df["price_gbp"].fillna(price_median)
    df["rating"] = df["rating"].fillna(rating_median).round().astype(int).clip(1, 5)
    df["price_inr"] = df["price_gbp"] * FIXED_GBP_TO_INR
    df["in_stock"] = df["in_stock"].astype(bool)

    keep = ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    return df[keep].reset_index(drop=True)


def create_database(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            );
            """
        )
        categories = sorted(df["category"].dropna().unique())
        conn.executemany("INSERT INTO categories(category_name) VALUES (?)", [(c,) for c in categories])
        cat_map = dict(conn.execute("SELECT category_name, category_id FROM categories").fetchall())
        rows = [
            (r.title, float(r.price_gbp), float(r.price_inr), int(r.rating), int(r.in_stock), cat_map[r.category])
            for r in df.itertuples(index=False)
        ]
        conn.executemany(
            "INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()


def main() -> None:
    raw = scrape_books()
    cleaned = clean_books(raw)
    cleaned.to_csv(OUT_DIR / "cleaned_books.csv", index=False)
    create_database(cleaned)
    print(f"Scraped and loaded {len(cleaned)} books across {cleaned['category'].nunique()} categories.")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
