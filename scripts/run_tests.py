#!/usr/bin/env python3
"""Run the test suite."""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    test_dir = Path(__file__).parent.parent / "tests"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"],
        cwd=str(Path(__file__).parent.parent),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
