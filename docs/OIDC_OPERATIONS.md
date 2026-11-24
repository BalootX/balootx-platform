# Operating OIDC sign-in

## Publish and activate a registry

1. Use the configuration admin API to create a draft with `POST /api/v1/configuration-registry/profiles`. Set `family` to `oidc_providers` and use global scope without rollout conditions. Include a reason for the change in the payload. The schema is available from `GET /api/v1/configuration-registry/profile-schemas/oidc_providers`.
2. Publish it with `POST /api/v1/configuration-registry/profiles/{revisionId}:publish` and the draft's `expectedEtag`. The configuration service retains published revisions, their payload digests, and the actor and time for creation and publication. The profile's events endpoint returns the audit chain.
3. Supply any referenced secrets from the deployment secret store. Set `BALOOT_OIDC_PROVIDERS_REVISION` to the published revision ID and restart the API. Choose a distinct profile ID if the deployment needs both revisions published during a rolling restart. Retire the old revision once it is no longer a rollback target.

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

Use HTTPS URLs throughout. Redirect URIs may use HTTP on localhost or loopback. A confidential client references a deployment environment variable with the `BALOOT_OIDC_CLIENT_SECRET` prefix. A public client omits `client_secret_env`. A missing referenced secret or an invalid, unpublished or unavailable revision stops API startup. Leave the registry unconfigured with no legacy variables, or publish an empty `providers` object, to disable OIDC sign-in.

## Migrate the previous release

1. Copy the issuer from `BALOOT_OIDC_ISSUER` byte for byte, including its trailing slash if present. Copy the provider key from `BALOOT_OIDC_PROVIDER_ID`. Use `default` when that variable was unset. Existing account links use both values.
2. Copy the authorization endpoint, token endpoint, client ID and redirect URI into the row. Set the provider's approved signing algorithms. A confidential client can reference the existing `BALOOT_OIDC_CLIENT_SECRET` deployment secret. The issuer now provides the signing-key URL through Discovery.
3. Keep the old variables until the previous release is no longer a rollback target. The new API ignores legacy provider metadata when a published registry revision is configured and logs one warning at startup.

## Configure MCP separately

MCP protected-resource metadata advertises the servers listed in `BALOOT_MCP_AUTHORIZATION_SERVERS`. List only authorization servers that issue tokens the MCP server accepts. MCP startup requires no OIDC provider credentials.
