-- ==============================================================================
-- File       : testing_practice.sql
-- Topic      : Day 48 — Test Database Isolation & Transaction Rollback Practice
-- Objective  : Demonstrate test schema setup, data isolation, and rollback testing in pure SQL.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- TASK 1: Create Test Isolated Tables Schema
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS test_users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS test_products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    stock INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS test_orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES test_users(id) ON DELETE CASCADE,
    total_amount NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending'
);

-- ------------------------------------------------------------------------------
-- TASK 2: Test Transaction Rollback (Simulating Out-Of-Stock Order Failure)
-- ------------------------------------------------------------------------------
-- Setup initial stock data
INSERT INTO test_products (name, price, stock) VALUES ('Laptop', 1000.00, 5);
INSERT INTO test_products (name, price, stock) VALUES ('Out-of-Stock GPU', 500.00, 0);

-- Begin test transaction
BEGIN TRANSACTION;

-- Step 1: Deduct Laptop stock (Success)
UPDATE test_products SET stock = stock - 1 WHERE name = 'Laptop';

-- Step 2: Attempt to deduct out-of-stock product (Simulating Failure Validation)
-- Stock is 0, so validation check fails!

-- Step 3: ROLLBACK test transaction to verify zero partial mutations occurred!
ROLLBACK;

-- Verify Laptop stock remained unchanged at 5 after rollback!
SELECT name, stock FROM test_products WHERE name = 'Laptop';

-- ------------------------------------------------------------------------------
-- TASK 3: Clean Test Fixture Teardown
-- ------------------------------------------------------------------------------
DELETE FROM test_orders;
DELETE FROM test_products;
DELETE FROM test_users;
