-- ==============================================================================
-- File       : observability_practice.sql
-- Topic      : Day 49 — Observability, System Event Logs & Database Readiness
-- Objective  : Demonstrate audit logging schema, request tracking, and SQL readiness probes.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- TASK 1: System Audit Log & Request Tracking Table Schema
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_audit_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    user_id INT NULL,
    method VARCHAR(10) NOT NULL,
    path VARCHAR(255) NOT NULL,
    status_code INT NOT NULL,
    duration_ms NUMERIC(10, 2) NOT NULL,
    event_message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_request_id ON system_audit_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_audit_status_code ON system_audit_logs(status_code);

-- ------------------------------------------------------------------------------
-- TASK 2: Database Readiness Probe Query (Simulates GET /health/ready check)
-- ------------------------------------------------------------------------------
-- Standard lightweight database connectivity probe query executed by application readiness handler
SELECT 1 AS readiness_status;

-- ------------------------------------------------------------------------------
-- TASK 3: Insert Simulated Request Logs & Query Slow Endpoints
-- ------------------------------------------------------------------------------
INSERT INTO system_audit_logs (request_id, user_id, method, path, status_code, duration_ms, event_message)
VALUES 
('req_8f92a1', 1, 'POST', '/auth/login', 200, 14.50, 'User authentication successful'),
('req_9b31c4', 2, 'POST', '/orders', 409, 28.10, 'Insufficient stock error for product #12'),
('req_7a12e9', NULL, 'GET', '/products', 200, 8.20, 'Product catalog listed');

-- Query all requests exceeding 20ms execution latency
SELECT request_id, method, path, status_code, duration_ms, event_message 
FROM system_audit_logs 
WHERE duration_ms > 20.00;
