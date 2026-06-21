"""Entry point so `python -m smart_run ...` works without installation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
