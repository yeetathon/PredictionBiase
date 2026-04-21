"""
DSG API authentication diagnostic — updated to show full response bodies.

Run from your Mac:
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

session = requests.Session()
session.headers["User-Agent"] = "AFL-Multi-Builder/2.0"
session.headers["Accept"] = "application/json"

AUTH_PARAMS = {"client": USERNAME, "authkey": AUTHKEY, "ftype": "json"}
AUTH_PARAMS_PW = {"client": USERNAME, "authkey": PASSWORD, "ftype": "json"}

basic_ak = base64.b64encode(f"{USERNAME}:{AUTHKEY}".encode()).decode()
basic_pw = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()

candidates = [
    # ── Correct path candidates (no /clients/ prefix) ─────────────────────
    ("australian_football/endpoint?client+authkey+ftype",
     f"{BASE}/australian_football/{ENDPOINT}", AUTH_PARAMS, {}),

    ("australian_football/endpoint?authkey+ftype only",
     f"{BASE}/australian_football/{ENDPOINT}", {"authkey": AUTHKEY, "ftype": "json"}, {}),

    ("australian_football/endpoint — password as authkey",
     f"{BASE}/australian_football/{ENDPOINT}", AUTH_PARAMS_PW, {}),

    ("afl/endpoint",
     f"{BASE}/afl/{ENDPOINT}", AUTH_PARAMS, {}),

    ("football/endpoint",
     f"{BASE}/football/{ENDPOINT}", AUTH_PARAMS, {}),

    # ── /clients/ prefix variants ─────────────────────────────────────────
    ("/clients/noahwal/australian_football — current pattern",
     f"{BASE}/clients/{USERNAME}/australian_football/{ENDPOINT}", AUTH_PARAMS, {}),

    ("/clients/noahwal/afl",
     f"{BASE}/clients/{USERNAME}/afl/{ENDPOINT}", AUTH_PARAMS, {}),

    # ── Basic Auth + no /clients/ ─────────────────────────────────────────
    ("No-prefix + Basic Auth username:authkey",
     f"{BASE}/australian_football/{ENDPOINT}", {"ftype": "json"},
     {"Authorization": f"Basic {basic_ak}"}),

    ("No-prefix + Basic Auth username:password",
     f"{BASE}/australian_football/{ENDPOINT}", {"ftype": "json"},
     {"Authorization": f"Basic {basic_pw}"}),

    # ── datasportsgroup.com variants ──────────────────────────────────────
    ("api.datasportsgroup.com",
     f"https://api.datasportsgroup.com/australian_football/{ENDPOINT}", AUTH_PARAMS, {}),

    ("datasportsgroup.com/clients/",
     f"https://datasportsgroup.com/clients/{USERNAME}/australian_football/{ENDPOINT}", AUTH_PARAMS, {}),
]

print("=" * 65)
print("DSG API authentication diagnostic")
print(f"Username : {USERNAME}")
print(f"Authkey  : {AUTHKEY[:8]}...")
print("=" * 65)

found = False
for label, url, params, headers in candidates:
    try:
        r = session.get(url, params=params, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "?")
        body = r.text[:300].strip().replace("\n", " ")
        print(f"\n[{r.status_code}] {label}")
        print(f"  URL : {r.url}")
        print(f"  CT  : {ct}")
        print(f"  BODY: {body}")
        if r.status_code == 200 and "html" not in ct.lower():
            print(f"\n  ✓ SUCCESS — this pattern works!")
            found = True
            break
    except Exception as exc:
        print(f"\n[ERR] {label}: {exc}")

if not found:
    print("\n" + "=" * 65)
    print("No pattern returned HTTP 200 non-HTML.")
    print("Please copy this entire output and share it.")
