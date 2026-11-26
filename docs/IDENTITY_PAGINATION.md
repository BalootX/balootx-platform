# Identity list pagination

`GET /api/v2/friends`, `GET /api/v2/friends/requests`, and `GET /api/v2/users:search` return `items` and `nextCursor`. Pass `nextCursor` as the `cursor` query parameter to fetch the next page. Stop when it is `null`. Reuse the same identity and endpoint for friend lists and the same query for user search. A cursor is opaque and can be used with a different page size.

Omit `limit` to use the default in the [bundled identity profile](../configs/identity/optional_simple_v1.yaml). Positive limits above the configured maximum are capped. Search treats wildcards as literal input and pages through the configured maximum result set. Each page reads current data.

## Release note

Existing `GET /api/v1/friends` and `GET /api/v1/friends/requests` keep their complete responses. SQLite pending requests retain `responded_at`. A valid service key can read its derived service identity or supply `x-baloot-subject-id` as before. V2 accepts the same service credentials and requires `baloot.read` for signed access tokens. The v2 friends list sorts by immutable subject ID so renames and guest upgrades cannot move a page boundary. The v2 pending requests use camelCase fields on both engines. Newest requests come first, with request ID breaking timestamp ties.

`GET /api/v1/users:search` retains its one-list response, guest access, wildcard matching, limit clamping, and sort by descending identity kind and display name. `GET /api/v2/users:search` requires `baloot.read` for signed access tokens. V2 denies guests by default, with an explicit opt-in in the bundled identity profile. Queries outside the configured length bounds, invalid cursors and limits below one return HTTP 422. V2 treats wildcards literally, ranks the exact username first and orders the remaining matches by subject ID inside the result cap.

A signed token with an unmapped role has no scopes on any policy-guarded route, including v1 routes that use the policy.

The Python SDK adds `friends_page`, `friend_requests`, and `search_users` with optional `cursor` and `limit` keywords. The TypeScript SDK adds `friendsPage`, `friendRequests`, and `searchUsers` with the same options. Existing `friends()` methods keep using v1. The Foundry client and direct HTTP callers use the [OpenAPI contract](../api/openapi.json).

## Migration notes

Run the [PostgreSQL migration runner](../scripts/migrate_postgres.py) before serving v2 pages. It builds the [friendship](../migrations/031_identity_list_pagination.sql) and [pending-request](../migrations/032_identity_pending_requests_index.sql) indexes concurrently. Each concurrent migration contains one schema-qualified index build. A later run repairs an invalid index left by an interrupted build. SQLite creates its indexes when the identity store opens. The hosted [integration check](../tests/test_v8_identity_sakka.py) records first-page and continuation plans before and after indexing. PostgreSQL documents [concurrent builds and interrupted indexes](https://www.postgresql.org/docs/17/sql-createindex.html).

After rolling back the service, reverse the PostgreSQL indexes if needed:

```sql
DROP INDEX CONCURRENTLY IF EXISTS identity.idx_identity_friendships_high_low;
DROP INDEX CONCURRENTLY IF EXISTS identity.idx_identity_pending_requests;
```
