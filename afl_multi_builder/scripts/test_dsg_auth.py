"""
DSG API credential verification — tests which sports are active on the account.
Run from your Mac:
    cd afl_multi_builder
    python scripts/test_dsg_auth.py
"""
import requests

USERNAME = "noahwal"
AUTHKEY  = "itylz4GZUpAEDHnV8mOrXPbQjxIMNTokWBL"
BASE     = "https://dsg-api.com"

session = requests.Session()
session.headers.update({
    "User-Agent": "AFL-Multi-Builder/2.0",
    "Accept": "application/json",
})

AUTH = {"client": USERNAME, "authkey": AUTHKEY, "ftype": "json"}

sports = [
    ("australian_football", "get_competitions"),
    ("soccer",              "get_competitions"),
    ("football",            "get_competitions"),
    ("aussie_rules",        "get_competitions"),
    ("afl",                 "get_competitions"),
    ("soccer",              "get_seasons"),
    ("soccer",              "get_matches"),
]

print("=" * 60)
print(f"DSG credential test — client={USERNAME}")
print("=" * 60)

for sport, endpoint in sports:
    url = f"{BASE}/{sport}/{endpoint}"
    try:
        r = session.get(url, params=AUTH, timeout=15)
        ct = r.headers.get("Content-Type", "")
        is_json = "json" in ct
        body = r.text[:120].strip().replace("\n", " ")
        icon = "✓" if (r.status_code == 200 and is_json) else " "
        print(f"  [{r.status_code}] {icon} {sport}/{endpoint}")
        if r.status_code != 200 or not is_json:
            print(f"         {body}")
        else:
            print(f"         {body}")
    except Exception as exc:
        print(f"  [ERR]   {sport}/{endpoint}: {exc}")
