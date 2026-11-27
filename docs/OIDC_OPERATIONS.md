# Operating OIDC sign-in

## Publish and activate a registry

1. Use the configuration admin API to create a draft with `POST /api/v1/configuration-registry/profiles`. Set `family` to `oidc_providers` and use global scope without rollout conditions. Include a reason for the change in the payload. The schema is available from `GET /api/v1/configuration-registry/profile-schemas/oidc_providers`.
2. Publish it with `POST /api/v1/configuration-registry/profiles/{revisionId}:publish` and the draft's `expectedEtag`. The configuration service retains published revisions, their payload digests, and the actor and time for creation and publication. The profile's events endpoint returns the audit chain.
3. Set `BALOOT_OIDC_PROVIDERS_PROFILE` to the profile ID. Configure any confidential registrations and supply their secrets from the deployment secret store. The HTTP API checks the published revision at startup and on refresh. An approved revision of the same profile activates without restarting the API. An empty registry disables all providers. A sign-in started against changed issuer or client metadata must start again. For a registry-only rollback, publish the previously approved payload under a new version. Registry publication preserves the migration rollback window.

Example request body

```json
{
  "family": "oidc_providers",
  "payload": {
    "schema_version": 1,
    "profile": {"id": "sign-in", "version": "1.0.0"},
    "change_reason": "Register the first sign-in provider",
    "providers": {
      "example": {
        "issuer": "https://issuer.example",
        "authorization_endpoint": "https://issuer.example/authorize",
        "token_endpoint": "https://issuer.example/token",
        "client_id": "example-client",
        "redirect_uri": "https://app.example/callback",
        "allowed_algorithms": ["RS256"],
        "scopes": ["openid", "profile", "email"],
        "clock_skew_seconds": 30,
        "http_timeout_seconds": 15,
        "discovery_cache_seconds": 300,
        "client_secret_env": "BALOOT_OIDC_CLIENT_SECRET_EXAMPLE"
      }
    }
  }
}
```

The keys in `providers` are the `providerId` values clients send. Each row declares which of `RS256`, `ES256` and `EdDSA` the provider may use to sign ID tokens, its OIDC scopes, tolerated clock skew, HTTP timeout, and Discovery cache lifetime. The client ID must appear in the token audience. The optional `trusted_additional_audiences` list defaults to empty. Multiple audiences require the token's `azp` to equal the client ID. If the token supplies an `azp` with a single audience, it must also equal the client ID. The API verifies the Discovery document's issuer exactly and uses its HTTPS `jwks_uri`. Each provider has its own client and key set. A changed signing-key URL takes effect on the next verification after the Discovery cache expires.

Use HTTPS URLs throughout. Redirect URIs may use HTTP on localhost or loopback. A public client omits `client_secret_env`. Leave the registry unconfigured with no legacy variables, or publish an empty `providers` object, to disable OIDC sign-in.

### Approve confidential registrations in the deployment

Set `BALOOT_OIDC_CREDENTIAL_BINDINGS` to a JSON array in the deployment environment. Each entry approves the exact provider key, issuer, client ID, full token endpoint URL and secret environment variable. The array contains references. Supply the secret values separately through the deployment secret store. Workers and MCP do not need these settings.

```json
[
  {
    "provider_id": "example",
    "issuer": "https://issuer.example",
    "client_id": "example-client",
    "token_endpoint": "https://issuer.example/token",
    "client_secret_env": "BALOOT_OIDC_CLIENT_SECRET_EXAMPLE"
  }
]
```

The API requires a matching entry before resolving a referenced credential. A missing binding, missing secret or invalid published revision stops HTTP startup. A rejected refresh makes sign-in temporarily unavailable until an approved revision is published or the deployment is corrected. To rotate to a new registration, endpoint or secret reference, deploy the new binding and secret alongside the current ones before publishing. Remove stale approvals after the rollback window closes. To change the value behind an unchanged reference, rotate it in the deployment secret store and restart the HTTP processes.

For PostgreSQL, use an application login subject to row-level security. The API checks the identity policies and the effective role before enabling sign-in. Use a schema owner for the schema migrations and a separate RLS-subject login for the issuer migration command.

The identity read policy limits regular reads to a caller's own account. Verified callbacks and the dedicated issuer migration login can look up links across accounts for their authorized operations. Both logins need `EXECUTE` on `security.oidc_identity_authorized()` when installation defaults restrict execution of new functions.

Signed-out clients can send JSON POSTs to the OIDC begin and callback routes without a session or API key. Clients presenting a valid session must follow the API's session CSRF policy. An API key or a verified access token follows the same rules as on other routes. A successful callback returns a session token in its response body. It does not set a browser cookie.

If the callback cannot save the account link or session, start a new sign-in when the response asks you to retry. For a link ownership conflict or incompatible provider configuration, contact support. The failed callback does not issue a session.

The current API restricts OIDC registry rows and audit history to authenticated configuration administrators on PostgreSQL and SQLite. Ordinary callers see only accessible profiles in listings. A direct lookup for a restricted revision returns not found. The sign-in runtime can resolve the published registry internally. PostgreSQL also keeps earlier API binaries from reading or modifying OIDC registry rows, including retained revisions. Use the current API for registry administration.

The sign-in path needs schema usage and the following table privileges in addition to any grants for other features.

| Tables | Privileges |
| --- | --- |
| `configuration.profile_revisions`, `configuration.invalidation_epochs` | `SELECT` |
| `security.oidc_transactions`, `security.external_identities` | `SELECT`, `INSERT`, `UPDATE` |
| `identity.subjects` | `SELECT`, `INSERT` |
| `identity.sessions` | `SELECT`, `INSERT`, `UPDATE` |

The issuer migration login needs usage on the `configuration` and `security` schemas and these table privileges.

| Tables | Privileges |
| --- | --- |
| `configuration.profile_revisions`, `configuration.aggregate_heads`, `configuration.invalidation_epochs`, `security.oidc_issuer_transitions` | `SELECT`, `INSERT`, `UPDATE` |
| `configuration.events` | `SELECT`, `INSERT` |
| `security.external_identities` | `SELECT`, `INSERT`, `UPDATE`, `DELETE` |
| `configuration.bundle_revisions` | `SELECT` on `revision_id`, `components`, `scheduled_predecessor_id`, `scheduled_successor_id` and `status` |

Rollback uses these bundle columns to check published and scheduled references before retiring a registry revision.

Reserve `UPDATE` on `security.oidc_issuer_transitions` for the issuer migration login. The identity policy uses this grant to distinguish it from the HTTP application login. Leave the `identity` schema and its session tables accessible only to the application and other roles that serve identity operations. The issuer migration command does its work in `configuration` and `security`.

For SQLite, stop the previous process before opening the databases with the new release. Keep the identity and security databases and their SQLite journals together on a local filesystem. Preserve the existing provider key and copy its issuer byte for byte, including any trailing slash. Issuer changes for linked providers require the PostgreSQL procedure below.

## Prepare a SQLite rollback

The predecessor reads ordinary configuration and its audit on a separate SQLite image. Keep the upgraded configuration database in a directory outside the predecessor's files and mounts for the whole rollback. Identity, sessions and provider links keep using their existing databases.

1. Stop all processes that can open the runtime databases. Move the upgraded configuration database and any remaining SQLite journal companions together into protected custody. Leave the identity and security databases and their journals in the runtime directory. Keep the predecessor's provider configuration for the unchanged issuer.
2. From the new release, prepare the predecessor image with the protected configuration as input. The output must be a new file without journal companions at the predecessor's normal configuration path.

   ```sh
   python -m baloot_platform.configuration.sqlite_rollback \
     --protected custody/configuration.sqlite3 --output runtime/configuration.sqlite3
   ```

3. Start the predecessor with the prepared runtime. Keep the protected input and its registry revisions and audit in custody. The image retains ordinary configuration and its audit. If preparation refuses the source because of linked configuration references, resolve custody before starting the predecessor.
4. To return to the new release, stop all predecessor processes and build a new image. This retains ordinary revisions and audit written during rollback and restores the original registry rows and audit from custody.

   ```sh
   python -m baloot_platform.configuration.sqlite_rollback \
     --protected custody/configuration.sqlite3 --rollback runtime/configuration.sqlite3 \
     --output custody/restored-configuration.sqlite3
   ```

   Move the stopped rollback database and its journal companions aside. Install the restored image as the runtime configuration database and start the new release. Keep both inputs until startup and audit verification pass.

## Migrate the previous release

The previous API cannot complete a PostgreSQL OIDC callback under RLS. Plan a sign-in pause for the cutover. The previous API can continue to serve other routes.

1. Deploy the new release alongside the serving release. Pause both OIDC routes at the ingress and let requests already executing finish before applying the migration. From the new release checkout, run `python scripts/migrate_postgres.py` against the PostgreSQL database. Apply the registry access policies before publishing the first OIDC profile.
2. Before starting the new API, run the new release's `baloot-migrate-oidc` command with the legacy provider environment, `BALOOT_OIDC_PROVIDERS_PROFILE`, and the PostgreSQL migration connection in `BALOOT_DATABASE_URL`. Use the approved signing algorithms for that provider. The command retains the legacy provider key, or `default` if it was unset. It reads the authoritative issuer from Discovery and requires the original issuer to match byte for byte apart from a trailing slash. For a confidential client, provide the deployment binding with that exact Discovery issuer, the legacy key and client ID, the full token endpoint and the `BALOOT_OIDC_CLIENT_SECRET` reference. The command checks the binding before publishing and re-keying existing links in the same transaction. Keep the returned revision ID.

   ```sh
   baloot-migrate-oidc --version 1.0.0 --algorithm RS256 --actor operator-id --reason "Migrate legacy provider"
   ```

3. Start the new HTTP API with the same profile ID, deployment binding and referenced confidential client secret. Route both OIDC endpoints to it once it is ready. The new callback accepts unconsumed legacy state while the transaction and authorization code are still valid. Keep the migrated provider key and client metadata stable until those transactions expire. A consumed or expired transaction requires a fresh sign-in. Keep the legacy environment variables until the rollback window closes. The new API logs one startup warning when it ignores those variables.

The migration warns when a legacy JWKS override is configured. The provider needs a valid Discovery document before migration. The transition mirrors link upserts. Direct link updates and deletes belong to the transition commands while aliases are active.

To roll back, pause new OIDC begins and keep callbacks on the new API with its registry and database intact. Allow outstanding transactions to complete or expire, then close the callback route and wait for active callbacks to finish. Run `baloot-migrate-oidc --rollback REVISION_ID --actor operator-id --reason "Restore previous release"` with the original migration revision ID. The command restores the legacy issuer keys, including links created during rollout, and retires the active revision of the migrated profile. Keep the registry access policies installed. Restore the previous API for other routes and keep OIDC paused until an API with RLS-capable callbacks is ready. The previous reader can decode the new state format, but the previous callback still lacks write access under RLS.

After accepting the new release, run `baloot-migrate-oidc --finalize REVISION_ID --actor operator-id --reason "Close rollback window"` to remove the legacy aliases, then remove the old metadata variables. Repeating a command preserves the original audit record.

## Configure MCP separately

MCP protected-resource metadata advertises the servers listed in `BALOOT_MCP_AUTHORIZATION_SERVERS`. List only authorization servers that issue tokens the MCP server accepts. MCP startup requires no OIDC provider credentials.
