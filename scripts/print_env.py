#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys


def main() -> None:
    print("python", sys.version.replace("\n", " "))
    print("platform", platform.platform())
    for name in ["tabpfn", "torch", "numpy", "scipy", "pandas", "sklearn"]:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "unknown"))


if __name__ == "__main__":
    main()
