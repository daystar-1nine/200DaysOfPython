# HTTP 404 Error Status Code and Web Server Routing

The HTTP 404 Not Found client error status response code indicates that the web server cannot locate the requested resource at the specified Uniform Resource Identifier (URI).
Unlike HTTP 401 Unauthorized or 403 Forbidden, a 404 indicates the endpoint or file path does not exist in the routing table.

In RESTful APIs, returning a 404 is appropriate when a requested entity ID does not exist in the underlying database.
In single-page applications (SPAs), web servers such as Nginx or Apache must rewrite missing asset paths to `index.html` (fallback routing) to enable client-side routers to handle internal navigation without returning 404 errors.