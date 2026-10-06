"""Start Voice Studio: `.venv/bin/python -m studio [--port 8800]`."""

import argparse

import uvicorn

from studio.server.main import create_app

parser = argparse.ArgumentParser(description="Voice Studio: AI video + Chatterbox voice replacement")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8800)
args = parser.parse_args()

uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
