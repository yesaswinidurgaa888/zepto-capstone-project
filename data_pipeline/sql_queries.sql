-- 1: SELECT + WHERE + ORDER BY
SELECT title, price_gbp FROM books WHERE price_gbp > 30 ORDER BY price_gbp DESC;

-- 2: ORDER BY + LIMIT
SELECT title, rating, price_gbp FROM books ORDER BY rating DESC, price_gbp DESC LIMIT 10;

-- 3: DISTINCT
SELECT DISTINCT category_name FROM categories ORDER BY category_name;

-- 4: BETWEEN
SELECT title, price_inr FROM books WHERE price_inr BETWEEN 1000 AND 2500 ORDER BY price_inr;

-- 5: JOIN
SELECT b.title, b.rating, b.price_inr, c.category_name
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC, c.category_name, b.title
LIMIT 10;
