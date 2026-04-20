"""
DEPRECATED — demo_loader.py has been removed.

This module previously provided demo/CSV-based data providers for development.
The system now operates exclusively on live data from real APIs.

If you are seeing this import in your code:
  - Remove any import of DemoAFLDataProvider, DemoOddsProvider, etc.
  - Use the real providers: AFLDataLoader, OddsAPIProvider
  - Configure AFL_DATA_AUTHKEY in your .env file

Any import from this module will raise ImportError immediately.
"""

raise ImportError(
    "demo_loader is no longer available. "
    "This system requires live data from real APIs.\n\n"
    "Required configuration:\n"
    "  AFL_DATA_AUTHKEY=<your_key>   (required)\n"
    "  ODDS_API_KEY=<your_key>       (optional — enables market edge)\n\n"
    "See README.md for setup instructions."
)
