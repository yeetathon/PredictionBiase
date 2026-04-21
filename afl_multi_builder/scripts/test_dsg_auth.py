"""
DSG API authentication diagnostic — phase 2.

The correct URL structure is confirmed:
  https://dsg-api.com/australian_football/{endpoint}?client=X&authkey=Y&ftype=json

This script now tries different client name variants and endpoints
to find the exact client identifier your account uses.

Run from your Mac:
    cd afl_multi_builder
    python scripts/test_dsg_auth.py
"""
import sys
import requests

USERNAME = "noahwal"
PASSWORD = "oKf4kn6r$R"
AUTHKEY  = "itylz4GZUpAEDHnV8mOrXPbQjxIMNTokWBL"
BASE     = "https://dsg-api.com"
SPORT    = "australian_football"

session = requests.Session()
session.headers.update({
    "User-Agent": "AFL-Multi-Builder/2.0",
    "Accept": "application/json",
})


def _try(label: str, endpoint: str, client: str, authkey: str) -> bool:
    url = f"{BASE}/{SPORT}/{endpoint}"
    params = {"client": client, "authkey": authkey, "ftype": "json"}
    try:
        r = session.get(url, params=params, timeout=15)
        ct = r.headers.get("Content-Type", "?")
        body = r.text[:300].strip().replace("\n", " ")
        is_json = "json" in ct.lower()
        ok = r.status_code == 200 and is_json
        icon = "✓" if ok else "✗"
        print(f"  [{r.status_code}] {icon} {label}")
        if r.status_code != 200 or not is_json:
            print(f"         {body}")
        if ok:
            print(f"         RESPONSE: {body}")
        return ok
    except Exception as exc:
        print(f"  [ERR]  {label}: {exc}")
        return False


# ── 1. Client name variants ──────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Client name variants (endpoint=get_competitions)")
print("=" * 65)

client_variants = [
    ("noahwal (current)",          "noahwal"),
    ("noahwallis",                 "noahwallis"),
    ("noah",                       "noah"),
    ("noah.wal",                   "noah.wal"),
    ("noah_wal",                   "noah_wal"),
    ("NOAHWAL (uppercase)",        "NOAHWAL"),
]

found_client = None
for label, client in client_variants:
    if _try(label, "get_competitions", client, AUTHKEY):
        found_client = client
        print(f"\n  ✓ SUCCESS: client='{client}' works!")
        break

# ── 2. Authkey variants ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 2 — Authkey variants (client=noahwal)")
print("=" * 65)

authkey_variants = [
    ("authkey as given",   AUTHKEY),
    ("password as authkey", PASSWORD),
]

found_authkey = None
for label, ak in authkey_variants:
    if _try(label, "get_competitions", "noahwal", ak):
        found_authkey = ak
        print(f"\n  ✓ SUCCESS: authkey='{ak[:8]}...' works!")
        break

# ── 3. Endpoint variants (maybe get_competitions not on your plan) ───────────
print()
print("=" * 65)
print("STEP 3 — Endpoint variants (client=noahwal, current authkey)")
print("=" * 65)

endpoints = [
    "get_competitions",
    "get_seasons",
    "get_matches",
    "get_rounds",
    "get_teams",
    "get_tables",
    "get_fixtures",
    "get_schedule",
]

found_endpoint = None
for ep in endpoints:
    if _try(ep, ep, "noahwal", AUTHKEY):
        found_endpoint = ep
        print(f"\n  ✓ SUCCESS: endpoint '{ep}' works!")
        break

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
if found_client or found_authkey or found_endpoint:
    print("DONE — share the ✓ line(s) above and the fix will be applied.")
else:
    print("All attempts returned 401.")
    print()
    print("Please log in to https://dsg-api.com and locate:")
    print("  • Your API 'client' identifier (may differ from login username)")
    print("  • Your API 'authkey' or 'API key' field")
    print("  • Which endpoints are listed under your AFL Advanced Pack")
    print()
    print("Copy the full output above and share it.")
