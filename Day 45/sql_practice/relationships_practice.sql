-- ==============================================================================
-- SQL Relationships Practice Queries (relationships_practice.sql)
-- Objective  : Pure SQL Foreign Keys, JOINs, Aggregations, and Many-to-Many junction tables.
-- Concept    : Database Relationships & Joins (Day 45 requirement)
-- ==============================================================================

-- 1. Create Users and Orders Tables (One-to-Many)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    amount NUMERIC(10, 2) NOT NULL
);

-- Insert Sample Data
INSERT INTO users (name, email) VALUES 
    ('Suraj', 'suraj@example.com'),
    ('Alex', 'alex@example.com');

INSERT INTO orders (user_id, amount) VALUES 
    (1, 500.00),
    (1, 1200.00),
    (1, 750.00),
    (2, 300.00);

-- Task 2: Query User Orders with JOIN
SELECT users.name, orders.amount 
FROM users 
JOIN orders ON users.id = orders.user_id;

-- Task 3: Calculate Total Spending (SUM & GROUP BY)
SELECT users.name, SUM(orders.amount) AS total_spent
FROM users
JOIN orders ON users.id = orders.user_id
GROUP BY users.id, users.name;

-- Task 7: Many-to-Many Relationship Tables
CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS student_courses (
    student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    PRIMARY KEY (student_id, course_id)
);

-- Insert Many-to-Many Data
INSERT INTO students (name) VALUES ('Suraj'), ('Alex');
INSERT INTO courses (title) VALUES ('Python'), ('SQL'), ('FastAPI'), ('Java');

INSERT INTO student_courses (student_id, course_id) VALUES 
    (1, 1), (1, 2), (1, 3), -- Suraj takes Python, SQL, FastAPI
    (2, 1), (2, 4);        -- Alex takes Python, Java

-- Task 7 Query 1: Find all courses taken by Suraj
SELECT courses.title 
FROM courses 
JOIN student_courses ON courses.id = student_courses.course_id
JOIN students ON students.id = student_courses.student_id
WHERE students.name = 'Suraj';

-- Task 7 Query 2: Find all students taking Python
SELECT students.name 
FROM students 
JOIN student_courses ON students.id = student_courses.student_id
JOIN courses ON courses.id = student_courses.course_id
WHERE courses.title = 'Python';

-- Task 7 Query 3: Count students in each course
SELECT courses.title, COUNT(student_courses.student_id) AS student_count
FROM courses
LEFT JOIN student_courses ON courses.id = student_courses.course_id
GROUP BY courses.id, courses.title;

-- Task 7 Query 4: Count courses for each student
SELECT students.name, COUNT(student_courses.course_id) AS course_count
FROM students
LEFT JOIN student_courses ON students.id = student_courses.student_id
GROUP BY students.id, students.name;
