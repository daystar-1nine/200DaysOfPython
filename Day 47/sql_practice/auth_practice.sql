-- ==============================================================================
-- File       : auth_practice.sql
-- Topic      : Day 47 — Authentication, Authorization & Security SQL Practice
-- Objective  : Master SQL schemas for authentication, password hashes, RBAC, and ownership queries.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- TASK 1: Create Users Table with Security Columns
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INT,
    phone VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast user authentication lookups by email
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_auth ON users(email);

-- ------------------------------------------------------------------------------
-- TASK 2: Insert User & Admin Records with Argon2/Bcrypt Password Hashes
-- ------------------------------------------------------------------------------
-- Regular User
INSERT INTO users (name, email, age, phone, password_hash, role)
VALUES (
    'Suraj Sawant',
    'suraj@example.com',
    21,
    '+91-9876543210',
    '$argon2id$v=19$m=65536,t=3,p=4$abc123hashvaluefortestingpassword',
    'user'
);

-- Admin User
INSERT INTO users (name, email, age, phone, password_hash, role)
VALUES (
    'System Administrator',
    'admin@example.com',
    30,
    '+91-9999999999',
    '$argon2id$v=19$m=65536,t=3,p=4$adminhashvaluefortestingpassword',
    'admin'
);

-- ------------------------------------------------------------------------------
-- TASK 3: Authentication Lookup Query (Email Login)
-- ------------------------------------------------------------------------------
-- Fetch user record by email to verify password hash
SELECT id, name, email, password_hash, role
FROM users
WHERE LOWER(email) = LOWER('suraj@example.com');

-- ------------------------------------------------------------------------------
-- TASK 4: Role-Based Authorization Query (Admin Users Endpoint)
-- ------------------------------------------------------------------------------
-- Fetch all users for Admin Dashboard (Role Check: role = 'admin')
SELECT id, name, email, role, created_at
FROM users
WHERE role = 'user' OR role = 'admin'
ORDER BY id ASC;

-- ------------------------------------------------------------------------------
-- TASK 5: Resource Ownership Enforcement Query (User's Own Orders)
-- ------------------------------------------------------------------------------
-- Fetch orders strictly belonging to the authenticated user ID (preventing ID tampering)
SELECT o.id AS order_id, o.total_amount, o.status, o.created_at,
       i.product_id, i.quantity, i.price
FROM orders o
JOIN order_items i ON o.id = i.order_id
WHERE o.user_id = 1  -- Bound to current_user.id from JWT
ORDER BY o.created_at DESC;
