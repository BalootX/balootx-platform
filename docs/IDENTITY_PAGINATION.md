# Identity list pagination

`GET /api/v1/friends`, `GET /api/v1/friends/requests`, and `GET /api/v1/users:search` return `items` and `nextCursor`. To fetch the next page, send the returned `nextCursor` as the `cursor` query parameter. Stop when `nextCursor` is `null`. Use the same identity and endpoint for friend lists and keep the same `query` for user search. A cursor is opaque and can be reused with a different page size.

Omit `limit` to use the default in the [identity pagination configuration](../configs/identity/pagination_v1.yaml). A positive `limit` above the configured maximum is capped. Invalid limits or cursors return HTTP 422. Each page reads current data. Changes to a user's sort fields or search visibility between requests can affect which results appear on later pages.

Friends are ordered by subject ID. Incoming pending requests are newest first, with request ID as the timestamp tie-breaker. User search keeps its kind and display-name ordering and adds subject ID as the final tie-breaker.

The Python SDK exposes `friends`, `friend_requests`, and `search_users` with optional `cursor` and `limit` keywords. The TypeScript SDK exposes `friends`, `friendRequests`, and `searchUsers` with the same options. See the [OpenAPI contract](../api/openapi.json) for direct HTTP clients.

For PostgreSQL, apply the [pagination index migration](../migrations/031_identity_list_pagination.sql) before serving pages. SQLite creates the indexes when its identity store opens. The hosted [integration check](../tests/test_v8_identity_sakka.py) records first-page and continuation query plans before and after the indexes on both engines. Substring matching in user search remains a filter on the ordered index scan.
