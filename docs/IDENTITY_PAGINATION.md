# Identity list pagination

`GET /api/v2/friends`, `GET /api/v2/friends/requests`, and `GET /api/v1/users:search` return `items` and `nextCursor`. Pass `nextCursor` as the `cursor` query parameter to fetch the next page. Stop when it is `null`. Reuse the same identity and endpoint for friend lists and the same query for user search. A cursor is opaque and can be used with a different page size.

Omit `limit` to use the default in the [identity pagination configuration](../configs/identity/pagination_v1.yaml). Positive limits above the configured maximum are capped. Invalid limits, cursors and short search queries return HTTP 422. Search requires a signed-in account or service identity by default. Operators can explicitly allow guest search in the same configuration. Search treats wildcards as literal input and pages only through the configured maximum result set. Each page reads current data.

## Release note

Existing `GET /api/v1/friends` and `GET /api/v1/friends/requests` continue to return the complete list. The v1 friends list retains display-name order. The v2 friends list sorts by immutable subject ID so renames and guest upgrades cannot move a page boundary and both database engines can use the friendship indexes. Pending requests use the same fields on both engines and are newest first, with request ID breaking timestamp ties. User search sorts by immutable subject ID and remembers only that ID in its continuation.

The Python SDK adds `friends_page`, `friend_requests`, and `search_users` with optional `cursor` and `limit` keywords. The TypeScript SDK adds `friendsPage`, `friendRequests`, and `searchUsers` with the same options. Existing `friends()` methods keep using v1. The Foundry client and direct HTTP callers use the [OpenAPI contract](../api/openapi.json).

## Migration notes

Apply the [PostgreSQL index migration](../migrations/031_identity_list_pagination.sql) before serving v2 pages. The transactional index build can briefly block identity writes, so schedule it for a deployment window. SQLite creates the indexes when the identity store opens. The hosted [integration check](../tests/test_v8_identity_sakka.py) prints first-page and continuation query plans on both engines, including a baseline taken before the new indexes. User search uses the existing subject primary-key index for the bounded result set.

After rolling back the service, reverse the PostgreSQL indexes if needed:

```sql
DROP INDEX IF EXISTS identity.idx_identity_friendships_high_low;
DROP INDEX IF EXISTS identity.idx_identity_pending_requests;
```
