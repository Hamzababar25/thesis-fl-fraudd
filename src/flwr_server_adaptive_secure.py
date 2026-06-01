"""Backward-compatible wrapper for renamed security FL server entrypoint.

Use `src/flwr_server_security.py` for all new commands.
"""

from flwr_server_security import main


if __name__ == "__main__":
    main()
