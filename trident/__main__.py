"""CLI entry point: `trident --config config/trident.yaml`."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .config import load_config
from .gateway import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="trident", description="TRIDENT inference router")
    parser.add_argument("--config", "-c", required=True, help="Path to trident YAML config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(load_config(args.config))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
