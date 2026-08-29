-- ==============================================================================
-- SQL Practice Queries (sql_practice.sql)
-- Objective  : Pure SQL CRUD, filtering, sorting, aggregations, and JOIN queries.
-- Concept    : SQL Relational Database Queries (Day 44 requirement)
-- ==============================================================================

-- Table DDL
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    amount NUMERIC(10, 2) NOT NULL
);

-- Query 1: Find all users older than 18
SELECT * FROM users WHERE age > 18;

-- Query 2: Find users whose name starts with S
SELECT * FROM users WHERE name LIKE 'S%';

-- Query 3: Sort users by age
SELECT * FROM users ORDER BY age ASC;

-- Query 4: Get the 5 oldest users
SELECT * FROM users ORDER BY age DESC LIMIT 5;

-- Query 5: Count users
SELECT COUNT(*) FROM users;

-- Query 6: Find all orders belonging to user 1
SELECT * FROM orders WHERE user_id = 1;

-- Query 7: Calculate total order amount
SELECT SUM(amount) AS total_amount FROM orders;

-- Query 8: Join users and orders
SELECT users.name, orders.amount 
FROM users 
JOIN orders ON users.id = orders.user_id;

-- Query 9: Find users who have orders
SELECT DISTINCT users.name 
FROM users 
JOIN orders ON users.id = orders.user_id;

-- Query 10: Find the highest-value order
SELECT * FROM orders ORDER BY amount DESC LIMIT 1;
