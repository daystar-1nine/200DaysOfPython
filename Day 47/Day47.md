# 🔐 Day 47 — Authentication & Authorization with FastAPI

## Executive Summary

Security is a mandatory foundation for production backend engineering. Everyday applications transition from open, unauthenticated access to secure systems where users register, authenticate using credentials, obtain cryptographic access tokens, and access protected resources according to their authorized roles and resource ownership.

Today's masterclass covers **Password Hashing** (Argon2 / Bcrypt), **JSON Web Tokens (JWT)**, **Bearer Authentication**, **Role-Based Access Control (RBAC)**, **Resource Ownership Enforcement**, **401 Unauthorized vs 403 Forbidden** status codes, and Alembic database schema migrations for security credentials.

---

## Key Technical Concepts

### 1. Authentication vs Authorization

```text
               CLIENT REQUEST
                      │
                      ▼
           1. AUTHENTICATION
            ("Who are you?")
         [Valid Credentials/JWT?]
           ├── NO  ──> 401 Unauthorized
           └── YES
                │
                ▼
           2. AUTHORIZATION
         ("What can you do?")
        [Role: Admin? Owner?]
           ├── NO  ──> 403 Forbidden
           └── YES ──> Allow Request
```

- **Authentication (AuthN):** Verifies the identity of a user (e.g., email + password verification, resulting in a signed JWT access token).
- **Authorization (AuthZ):** Determines whether an authenticated user has permission to access a specific resource or endpoint (e.g., admin role check, resource ownership verification).

---

### 2. Password Hashing vs Encryption

> [!CAUTION]
> Never store plaintext passwords in a database! If database backup files or table contents are leaked, all user accounts become exposed.

- **Encryption (Two-Way):** Converts plaintext to ciphertext using a key, which can later be decrypted back to plaintext. Not suitable for passwords.
- **Hashing (One-Way):** Computes a fixed-length cryptographic hash from plaintext input. Password hashes cannot be decrypted. During authentication, the candidate password is hashed and compared against the stored hash.

#### Recommended Hashing Algorithms:
1. **Argon2id (Winner of Password Hashing Competition):** Memory-hard and time-hard algorithm resistant to GPU/ASIC brute-force attacks (`pwdlib[argon2]`).
2. **Bcrypt:** Standard, time-tested salted hashing algorithm (`passlib[bcrypt]`).

---

### 3. JSON Web Tokens (JWT) & Bearer Authentication

A **JWT** is an open standard (RFC 7519) for securely transmitting information between parties as a JSON object.

A JWT consists of three parts separated by dots (`.`):
$$\text{JWT} = \text{Header} . \text{Payload} . \text{Signature}$$

```text
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzA5MzE4NDAwfQ.SignatureString
```

1. **Header:** Algorithm and token type (`{"alg": "HS256", "typ": "JWT"}`).
2. **Payload:** Claims containing non-sensitive user metadata and expiration (`{"sub": "1", "role": "user", "exp": 1709318400}`).
3. **Signature:** Cryptographic hash generated using `HMAC-SHA256(Header + Payload, SECRET_KEY)` to prevent client-side token tampering.

> [!WARNING]
> JWT payloads are Base64URL-encoded and publicly readable! **Never** place plaintext passwords, credit card numbers, or secret keys inside a JWT payload.

---

### 4. Status Codes: 401 Unauthorized vs 403 Forbidden

| HTTP Status Code | Name | Meaning | Common Triggers |
| :--- | :--- | :--- | :--- |
| **`401`** | **Unauthorized** | Missing, invalid, or expired authentication token. | No `Authorization` header, invalid JWT signature, expired token. |
| **`403`** | **Forbidden** | User is authenticated, but lacks required permissions. | Regular user attempting `POST /products` (admin-only) or accessing another user's order. |

---

### 5. Resource Ownership Validation

> [!IMPORTANT]
> Never trust client-supplied `user_id` query parameters (e.g., `GET /orders?user_id=2`) for authorization! A user could modify the URL to access another user's private data.

**Correct Ownership Pattern:**
1. Extract `user_id` directly from the validated JWT token (`get_current_user`).
2. Query the database filtering by `WHERE user_id = current_user.id`.

---

## ❓ Practice & Interview Q&A (All 20 Questions)

### Q1: What is authentication?
**Answer:** Authentication is the process of verifying a user's identity using credentials such as an email and password or cryptographic token.

### Q2: What is authorization?
**Answer:** Authorization is the process of determining whether an authenticated user has permission to perform an action or access a specific resource.

### Q3: What is the main difference between authentication and authorization?
**Answer:** Authentication asks *"Who are you?"*, while authorization asks *"What are you allowed to do?"*.

### Q4: Why shouldn't passwords be stored directly in plain text?
**Answer:** Storing plaintext passwords exposes user accounts to severe compromise if database backups, logs, or SQL injection vulnerabilities occur.

### Q5: What is password hashing?
**Answer:** Password hashing is a one-way cryptographic function that transforms a plaintext password into an irreversible hash digest using random salts.

### Q6: What is the difference between hashing and encryption?
**Answer:** Encryption is two-way and can be decrypted using a secret key. Hashing is one-way and cannot be decrypted back to plaintext.

### Q7: What is a JWT (JSON Web Token)?
**Answer:** A JWT is a compact, URL-safe token format representing claims signed cryptographically between a client and server.

### Q8: What are the three parts of a JWT?
**Answer:** Header (algorithm/type), Payload (user claims & expiration), and Signature (HMAC signature proving token authenticity).

### Q9: What is an access token?
**Answer:** An access token is a short-lived token provided upon login that client applications attach to HTTP headers to access protected API endpoints.

### Q10: What is a Bearer token?
**Answer:** Bearer authentication is an HTTP authentication scheme where the client sends `Authorization: Bearer <token>` in request headers.

### Q11: Why should JWTs expire?
**Answer:** Expiration (`exp` claim) limits the window of opportunity for an attacker to reuse a stolen or intercepted access token.

### Q12: What does HTTP 401 Unauthorized mean?
**Answer:** HTTP 401 indicates that the request lacks valid authentication credentials (missing, invalid, or expired token).

### Q13: What does HTTP 403 Forbidden mean?
**Answer:** HTTP 403 indicates that the user is authenticated, but does not possess the required permissions or roles for the resource.

### Q14: How does FastAPI dependency injection help with authentication?
**Answer:** FastAPI dependencies (e.g., `Depends(get_current_user)`) automatically intercept HTTP requests, extract Bearer headers, decode JWTs, fetch user models, and enforce security before executing route handlers.

### Q15: How would you protect an API endpoint in FastAPI?
**Answer:** Include `current_user: User = Depends(get_current_user)` in the route handler signature.

### Q16: How would you implement admin-only access?
**Answer:** Create an admin dependency `require_admin(user: User = Depends(get_current_user))` that checks if `user.role == "admin"`, raising HTTP 403 Forbidden if false.

### Q17: How do you ensure a user can only see their own orders?
**Answer:** Derive `user_id` strictly from `current_user.id` obtained from the JWT token, and filter database queries with `.where(Order.user_id == current_user.id)`.

### Q18: Why shouldn't JWT payloads contain sensitive data like passwords?
**Answer:** JWT payloads are Base64URL-encoded strings that anyone can decode and read without needing the secret key.

### Q19: Where should JWT secret keys be stored?
**Answer:** Secret keys must be stored in secure environment variables (`.env`) outside version control repositories.

### Q20: What is the difference between an access token and a refresh token?
**Answer:** An access token is short-lived (e.g., 30 minutes) for API requests. A refresh token is long-lived (e.g., 7 days) and used exclusively to request a new access token without re-entering credentials.

---

## 📋 Day 47 Recap Checklist

- [x] Implemented password hashing with `pwdlib` (Argon2) / `passlib` (bcrypt).
- [x] Built JWT token generation and decoding in `app/security.py`.
- [x] Created `POST /auth/register` and `POST /auth/login` endpoints.
- [x] Implemented `get_current_user` and `require_admin` FastAPI dependencies.
- [x] Extended User model and ran Alembic migration `007_add_auth_fields.py`.
- [x] Enforced RBAC (Admin-only `/products` creation/deletion and `/admin/users`).
- [x] Enforced resource ownership on `/users/me/orders` and `GET /orders/{id}`.
- [x] Verified 35+ Pytest test cases across all endpoints.
