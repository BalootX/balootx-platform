from .platform_v19 import build_platform_v19_router
from .configuration import build_configuration_router
from .hitl import build_hitl_router
from .adjudication import build_adjudication_router
from .auth import build_auth_router
from .canonical_history import build_canonical_history_router
from .live import build_live_router
from .sakka import build_sakka_router
from .foundry_admin_v2 import build_foundry_admin_v2_router
from .security import build_security_router
from .shared import build_shared_router

__all__ = [
    "build_adjudication_router", "build_auth_router", "build_canonical_history_router", "build_live_router",
    "build_sakka_router", "build_security_router", "build_shared_router", "build_hitl_router", "build_configuration_router", "build_platform_v19_router", "build_foundry_admin_v2_router",
]
