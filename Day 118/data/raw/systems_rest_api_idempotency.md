# REST API Idempotency and HTTP Method Semantics

An HTTP method is considered idempotent if executing identical requests multiple times produces the exact same server-side state as executing it a single time.
`GET`, `HEAD`, `OPTIONS`, `PUT`, and `DELETE` are defined as idempotent methods under the HTTP specification.
`POST` and `PATCH` are typically non-idempotent because repeated submissions can generate duplicate entities or side-effects.

To make `POST` transactions idempotent in payment and billing APIs, clients transmit unique `Idempotency-Key` headers stored in an in-memory cache like Redis.