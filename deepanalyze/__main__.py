"""CLI entrypoint for `python3 -m deepanalyze`."""

import sys
from .server import cli_entrypoint
from .wizard import AirGapWizard

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("server", "start", "--help", "-h"):
        cli_entrypoint()
    else:
        wizard = AirGapWizard()
        wizard.run()
