-- ==============================================================================
-- File       : migrations_practice.sql
-- Topic      : Day 46 — Database Migrations, ALTER TABLE, and Indexing Practice
-- Objective  : Master raw DDL migration statements and index query plan analysis.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- TASK 1: Base Tables Creation (Initial Schema Revision 001)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    stock INT NOT NULL DEFAULT 0,
    description TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    total_amount NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id INT NOT NULL REFERENCES products(id),
    quantity INT NOT NULL DEFAULT 1,
    price NUMERIC(10, 2) NOT NULL
);

-- ------------------------------------------------------------------------------
-- TASK 2: Migration 002 — Add Phone Column to Users
-- ------------------------------------------------------------------------------
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Rollback for Task 2:
-- ALTER TABLE users DROP COLUMN phone;

-- ------------------------------------------------------------------------------
-- TASK 3: Migration 003 — Add Category Column to Products
-- ------------------------------------------------------------------------------
ALTER TABLE products ADD COLUMN category VARCHAR(50);

-- Rollback for Task 3:
-- ALTER TABLE products DROP COLUMN category;

-- ------------------------------------------------------------------------------
-- TASK 4: Migration 004 — Add Created At Column to Products
-- ------------------------------------------------------------------------------
ALTER TABLE products ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- ------------------------------------------------------------------------------
-- TASK 5: Migration 005 — Add Indexes for Rapid Lookups
-- ------------------------------------------------------------------------------
-- Create index on users(email) for fast authentication lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Create index on products(name) for fast catalog search
CREATE INDEX IF NOT EXISTS idx_products_name ON products(name);

-- ------------------------------------------------------------------------------
-- TASK 6: Migration 006 — Add Created At Column to Users (Data Migration Scenario)
-- Step A: Add column allowing NULL initially
ALTER TABLE users ADD COLUMN created_at TIMESTAMP WITH TIME ZONE;

-- Step B: Populate existing rows with default timestamp
UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL;

-- Step C: Set default constraint for future inserts
ALTER TABLE users ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;

-- ------------------------------------------------------------------------------
-- TASK 7: Query Plan Inspection (EXPLAIN & EXPLAIN ANALYZE)
-- ------------------------------------------------------------------------------
-- Inspect query plan utilizing index on email lookup
EXPLAIN SELECT * FROM users WHERE email = 'suraj@example.com';

-- Inspect query plan for product search
EXPLAIN SELECT * FROM products WHERE name ILIKE '%keyboard%';
