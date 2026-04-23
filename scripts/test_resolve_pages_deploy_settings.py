from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module():
    path = Path("scripts/resolve_pages_deploy_settings.py")
    spec = spec_from_file_location("resolve_pages_deploy_settings", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_module()
    direct_seed = "/ip4/3.149.166.58/udp/443/webrtc-direct/certhash/uEiBVooCaHXutTKs8lRU1X09zYELZl49YF8f9WGCd85c8gg"

    def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("explicit Pages seeds should not require network discovery")

    module.fetch_signed_seed_advertisement = unexpected_fetch
    module.fetch_browser_edge_snapshot = unexpected_fetch

    resolved = module.resolve_seed_node_urls(
        "https://edge.dragon.aberration.technology",
        "",
        direct_seed,
    )
    assert resolved == [direct_seed], resolved
    print("resolve-pages-deploy-settings-ok")


if __name__ == "__main__":
    main()
