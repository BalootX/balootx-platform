# Operating OIDC sign-in

## Publish and activate a registry

1. Use the configuration admin API to create a draft with `POST /api/v1/configuration-registry/profiles`. Set `family` to `oidc_providers` and use global scope without rollout conditions. Include a reason for the change in the payload. The schema is available from `GET /api/v1/configuration-registry/profile-schemas/oidc_providers`.
2. Publish it with `POST /api/v1/configuration-registry/profiles/{revisionId}:publish` and the draft's `expectedEtag`. The configuration service retains published revisions, their payload digests, and the actor and time for creation and publication. The profile's events endpoint returns the audit chain.
3. Set `BALOOT_OIDC_PROVIDERS_PROFILE` to the profile ID and supply referenced secrets from the deployment secret store. Only HTTP API processes need this setting and the credentials. At startup each API instance resolves the current published revision and logs its revision ID. Publish a new revision of the same profile, then roll the API to activate it. For a registry-only rollback, publish the previously approved payload under a new version of the same profile.

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
        "client_secret_env": "BALOOT_OIDC_CLIENT_SECRET_EXAMPLE"
      }
    }
  }
}
```

The keys in `providers` are the `providerId` values clients send. Each row declares which of `RS256`, `ES256` and `EdDSA` that provider may use to sign ID tokens. The API verifies the Discovery document's issuer exactly, then uses its HTTPS `jwks_uri`. Each provider has its own cached client and key set. Restart the API when a provider changes its signing-key URL.

Use HTTPS URLs throughout. Redirect URIs may use HTTP on localhost or loopback. A confidential client references a deployment environment variable with the `BALOOT_OIDC_CLIENT_SECRET` prefix. A public client omits `client_secret_env`. A missing secret or a configured profile without a valid published revision stops HTTP startup. Leave the registry unconfigured with no legacy variables, or publish an empty `providers` object, to disable OIDC sign-in.

## Migrate the previous release

1. Deploy the new release alongside the serving release. From the new release checkout, run `python scripts/migrate_postgres.py` against the PostgreSQL database. Keep the previous API serving sign-in throughout these steps.
2. Before starting the new API, run the new release's `baloot-migrate-oidc` command with the legacy provider environment, `BALOOT_OIDC_PROVIDERS_PROFILE`, and the PostgreSQL migration connection in `BALOOT_DATABASE_URL`. Use the approved signing algorithms for that provider. The command retains the legacy provider key, or `default` if it was unset. It reads the authoritative issuer from Discovery and requires the original issuer to match byte for byte apart from a trailing slash. When the issuer changes, the same transaction publishes the audited profile and re-keys existing links. Both issuer keys resolve to the same account while old and new API instances overlap. Keep the returned revision ID.

   ```sh
   baloot-migrate-oidc --version 1.0.0 --algorithm RS256 --actor operator-id --reason "Migrate legacy provider"
   ```

3. Start the new HTTP API with the same profile ID and the referenced confidential client secret. Move traffic after it is ready, then drain the previous release. Keep the legacy environment variables until the rollback window closes. The new API logs one startup warning when it ignores those variables.

To roll back, restore the previous API with its legacy variables and move traffic back to it. After draining the new API, run `baloot-migrate-oidc --rollback REVISION_ID --actor operator-id --reason "Restore previous release"`. This restores the legacy issuer keys, including links created during rollout, and retires that migration's profile revision. After closing the rollback window, run `baloot-migrate-oidc --finalize REVISION_ID --actor operator-id --reason "Close rollback window"` to remove the legacy aliases, then remove the old metadata variables. Each command can be repeated safely with the same inputs.

## Configure MCP separately

MCP protected-resource metadata advertises the servers listed in `BALOOT_MCP_AUTHORIZATION_SERVERS`. List only authorization servers that issue tokens the MCP server accepts. MCP startup requires no OIDC provider credentials.
