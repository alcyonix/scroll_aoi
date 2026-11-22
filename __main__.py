"""Entry point: `python -m aoi_tool <command> ...`

The network guard is installed on the very first lines, before numpy, OpenCV
or ONNX Runtime are imported anywhere in the process.
"""

import sys

from aoi_tool import netguard

netguard.install(allow_subprocess="--allow-subprocess" in sys.argv)

from aoi_tool.cli import main  # noqa: E402  (must come after the guard)

if __name__ == "__main__":
    sys.exit(main())
