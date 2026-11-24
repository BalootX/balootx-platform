# Operating OIDC sign-in

## Publish and activate a registry

1. Use the configuration admin API to create a draft with `POST /api/v1/configuration-registry/profiles`. Set `family` to `oidc_providers` and use global scope without rollout conditions. Include a reason for the change in the payload. The schema is available from `GET /api/v1/configuration-registry/profile-schemas/oidc_providers`.
2. Publish it with `POST /api/v1/configuration-registry/profiles/{revisionId}:publish` and the draft's `expectedEtag`. The configuration service retains published revisions, their payload digests, and the actor and time for creation and publication. The profile's events endpoint returns the audit chain.
3. Set `BALOOT_OIDC_PROVIDERS_PROFILE` to the profile ID and supply referenced secrets from the deployment secret store. Only HTTP API processes need this setting and the credentials. An API instance loads the published revision at startup and refreshes it on sign-in requests. Publishing a revision of the same profile activates it without restarting the API. An empty registry disables all providers. A sign-in started against changed issuer or client metadata must start again. For a registry-only rollback, publish the previously approved payload under a new version. Registry publication preserves the migration rollback window.

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

Use HTTPS URLs throughout. Redirect URIs may use HTTP on localhost or loopback. A confidential client references a deployment environment variable with the `BALOOT_OIDC_CLIENT_SECRET` prefix. A public client omits `client_secret_env`. A missing secret or a configured profile without a valid published revision stops HTTP startup. Leave the registry unconfigured with no legacy variables, or publish an empty `providers` object, to disable OIDC sign-in.

For PostgreSQL, use an application role subject to row-level security. The API checks that RLS applies to its effective role on `security.external_identities` before enabling sign-in. Use a separate migration role to apply schema and data changes.

The sign-in path needs schema usage and the following table privileges in addition to any grants for other features.

| Tables | Privileges |
| --- | --- |
| `configuration.profile_revisions`, `configuration.invalidation_epochs` | `SELECT` |
| `security.oidc_transactions`, `security.external_identities` | `SELECT`, `INSERT`, `UPDATE` |
| `identity.subjects` | `SELECT`, `INSERT` |
| `identity.sessions` | `INSERT` |

## Migrate the previous release

The previous API cannot complete a PostgreSQL OIDC callback under RLS. Plan a sign-in pause for the cutover. The previous API can continue to serve other routes.

1. Deploy the new release alongside the serving release. Pause both OIDC routes at the ingress and let requests already executing finish before applying the migration. From the new release checkout, run `python scripts/migrate_postgres.py` against the PostgreSQL database.
2. Before starting the new API, run the new release's `baloot-migrate-oidc` command with the legacy provider environment, `BALOOT_OIDC_PROVIDERS_PROFILE`, and the PostgreSQL migration connection in `BALOOT_DATABASE_URL`. Use the approved signing algorithms for that provider. The command retains the legacy provider key, or `default` if it was unset. It reads the authoritative issuer from Discovery and requires the original issuer to match byte for byte apart from a trailing slash. When the issuer changes, the same transaction publishes the audited profile and re-keys existing links. Keep the returned revision ID.

   ```sh
   baloot-migrate-oidc --version 1.0.0 --algorithm RS256 --actor operator-id --reason "Migrate legacy provider"
   ```

3. Start the new HTTP API with the same profile ID and the referenced confidential client secret. Route both OIDC endpoints to it once it is ready. The new callback accepts unconsumed legacy state while the transaction and authorization code are still valid. Keep the migrated provider key and client metadata stable until those transactions expire. A consumed or expired transaction requires a fresh sign-in. Keep the legacy environment variables until the rollback window closes. The new API logs one startup warning when it ignores those variables.

The migration warns when a legacy JWKS override is configured. The provider needs a valid Discovery document before migration. The transition mirrors link upserts. Direct link updates and deletes belong to the transition commands while aliases are active.

To roll back, pause new OIDC begins and keep callbacks on the new API with its registry and database intact. Allow outstanding transactions to complete or expire, then close the callback route and wait for active callbacks to finish. Run `baloot-migrate-oidc --rollback REVISION_ID --actor operator-id --reason "Restore previous release"` with the original migration revision ID. The command restores the legacy issuer keys, including links created during rollout, and retires the active revision of the migrated profile. Restore the previous API for other routes and keep OIDC paused until an API with RLS-capable callbacks is ready. The previous reader can decode the new state format, but the previous callback still lacks write access under RLS.

After accepting the new release, run `baloot-migrate-oidc --finalize REVISION_ID --actor operator-id --reason "Close rollback window"` to remove the legacy aliases, then remove the old metadata variables. Repeating a command preserves the original audit record.

## Configure MCP separately

MCP protected-resource metadata advertises the servers listed in `BALOOT_MCP_AUTHORIZATION_SERVERS`. List only authorization servers that issue tokens the MCP server accepts. MCP startup requires no OIDC provider credentials.
