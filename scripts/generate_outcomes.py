"""Compatibility entrypoint; the installed command is `baloot generate`."""
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from baloot_platform.research.outcome_cli import main as generate
    generate()


if __name__ == '__main__':
    main()
