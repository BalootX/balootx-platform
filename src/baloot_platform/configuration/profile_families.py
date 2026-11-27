from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from baloot_platform.security.oidc import (
    OIDC_PROVIDER_FAMILY,
    OidcProviderProfile,
    validate_oidc_provider_profile,
)


@dataclass(frozen=True)
class ProfileFamily:
    validate: Callable[[dict[str, Any]], Any]
    schema: Callable[[], dict[str, Any]]
    requires_config_admin: bool = False
    in_bundle: bool = True


PROFILE_FAMILY_DEFINITIONS = {
    OIDC_PROVIDER_FAMILY: ProfileFamily(
        validate_oidc_provider_profile,
        OidcProviderProfile.model_json_schema,
        requires_config_admin=True,
        in_bundle=False,
    )
}
