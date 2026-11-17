import faulthandler
import os
from pathlib import Path


def run():
    if os.environ.get("KV_SSD_NATIVE_ENV") == "1" and (
        path := os.environ.get("KV_SSD_METADATA_CACHE")
    ):
        from .metadata_cache import install

        install(Path(path))

    from .cli import main

    if seconds := int(os.environ.get("KV_SSD_DEBUG_STACK_SECONDS", "0")):
        faulthandler.dump_traceback_later(seconds, repeat=True)
    raise SystemExit(main())


run()
