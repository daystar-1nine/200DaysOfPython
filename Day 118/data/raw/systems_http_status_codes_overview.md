# HTTP Status Code Classes and Error Handling

HTTP response status codes are grouped into five categorical classes:
1. 1xx (Informational): Request received, continuing process (e.g., 101 Switching Protocols).
2. 2xx (Success): Action successfully received, understood, and accepted (e.g., 200 OK, 201 Created, 204 No Content).
3. 3xx (Redirection): Further action required to complete request (e.g., 301 Moved Permanently, 304 Not Modified).
4. 4xx (Client Error): Request contains bad syntax or cannot be fulfilled (e.g., 400 Bad Request, 403 Forbidden, 404 Not Found, 429 Too Many Requests).
5. 5xx (Server Error): Server failed to fulfill an apparently valid request (e.g., 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable).