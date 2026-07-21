"""Allow ``python -m nexus`` to invoke the public CLI."""

from nexus.api import main

raise SystemExit(main())
