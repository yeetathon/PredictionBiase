"""
DSG API authentication diagnostic.

Run this from your Mac to identify the correct endpoint URL and auth method.
It tries several patterns and reports which one succeeds (HTTP 200).

Usage:
    cd afl_multi_builder
    python scripts/test_dsg_auth.py
"""
import base64
import sys
import requests

USERNAME = "noahwal"
PASSWORD = "oKf4kn6r$R"
AUTHKEY  = "itylz4GZUpAEDHnV8mOrXPbQjxIMNTokWBL"
BASE     = "https://dsg-api.com"

ENDPOINT = "get_competitions"
SPORT    = "australian_football"

session = requests.Session()
session.headers["User-Agent"] = "AFL-Multi-Builder/2.0"

def _try(label: str, url: str, params: dict, headers: dict = {}) -> bool:
    try:
        r = session.get(url, params=params, headers=headers, timeout=15)
        body = r.text[:200].strip()
        print(f"  [{r.status_code}] {label}")
        if r.status_code == 200:
            print(f"         ✓  RESPONSE: {body}")
            return True
        else:
            print(f"         ✗  BODY: {body}")
    except Exception as exc:
        print(f"  [ERR] {label}: {exc}")
    return False

candidates = []

# ── Pattern 1: /clients/{client}/sport/endpoint?client=&authkey=&ftype=json ──
candidates.append((
    "PATH=clients/noahwal, params: client+authkey+ftype",
    f"{BASE}/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"client": USERNAME, "authkey": AUTHKEY, "ftype": "json"},
    {},
))

# ── Pattern 2: Same path, authkey only (no client in query) ──
candidates.append((
    "PATH=clients/noahwal, params: authkey+ftype only",
    f"{BASE}/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"authkey": AUTHKEY, "ftype": "json"},
    {},
))

# ── Pattern 3: Same path, password as authkey ──
candidates.append((
    "PATH=clients/noahwal, params: password-as-authkey+ftype",
    f"{BASE}/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"authkey": PASSWORD, "ftype": "json"},
    {},
))

# ── Pattern 4: No /clients/ prefix ──
candidates.append((
    "No /clients/, params: client+authkey+ftype",
    f"{BASE}/{SPORT}/{ENDPOINT}",
    {"client": USERNAME, "authkey": AUTHKEY, "ftype": "json"},
    {},
))

# ── Pattern 5: Basic Auth (username:authkey) ──
basic_ak = base64.b64encode(f"{USERNAME}:{AUTHKEY}".encode()).decode()
candidates.append((
    "Basic Auth username:AUTHKEY",
    f"{BASE}/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"ftype": "json"},
    {"Authorization": f"Basic {basic_ak}"},
))

# ── Pattern 6: Basic Auth (username:password) ──
basic_pw = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
candidates.append((
    "Basic Auth username:PASSWORD",
    f"{BASE}/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"ftype": "json"},
    {"Authorization": f"Basic {basic_pw}"},
))

# ── Pattern 7: /api/ prefix ──
candidates.append((
    "PATH=/api/clients/noahwal, params: client+authkey+ftype",
    f"{BASE}/api/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"client": USERNAME, "authkey": AUTHKEY, "ftype": "json"},
    {},
))

# ── Pattern 8: v1 prefix ──
candidates.append((
    "PATH=/v1/clients/noahwal",
    f"{BASE}/v1/clients/{USERNAME}/{SPORT}/{ENDPOINT}",
    {"client": USERNAME, "authkey": AUTHKEY, "ftype": "json"},
    {},
))

print("=" * 60)
print("DSG API authentication diagnostic")
print(f"Username : {USERNAME}")
print(f"Authkey  : {AUTHKEY[:8]}...")
print("=" * 60)

found = False
for label, url, params, headers in candidates:
    if _try(label, url, params, headers):
        print()
        print("SUCCESS! Working pattern:")
        print(f"  URL    : {url}")
        print(f"  Params : {params}")
        print(f"  Headers: {headers}")
        found = True
        break

if not found:
    print()
    print("No pattern returned HTTP 200.")
    print("Please log in to https://dsg-api.com and check:")
    print("  1. Your API client name (may differ from your login username)")
    print("  2. Your authkey (from your account dashboard)")
    print("  3. Whether your account has the AFL Advanced Pack enabled")
    sys.exit(1)
