"""Candidate-span proposer for chain-of-options.

Proposes typed spans (datetime, duration, timezone, amount, percent, number)
found in a state. It never decides anything: every span is offered as an
option to a Von Choice, and the model does the binding. Regex here only
answers "what could this slot refer to?", never "what is the answer?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9

# Named zones -> fixed UTC offsets in hours (DST names carry their own offset,
# which is exactly what these items hinge on).
ZONES: Dict[str, float] = {
    "utc": 0, "gmt": 0, "z": 0,
    "cet": 1, "cest": 2, "wet": 0, "west": 1, "eet": 2, "eest": 3, "bst": 1, "ist": 5.5,
    "est": -5, "edt": -4, "cst": -6, "cdt": -5, "mst": -7, "mdt": -6, "pst": -8, "pdt": -7,
    "jst": 9, "kst": 9, "aest": 10, "aedt": 11, "hkt": 8, "sgt": 8,
}
CITY_ZONES: Dict[str, str] = {
    "central european": "Europe/Amsterdam", "eastern european": "Europe/Athens", "western european": "Europe/Lisbon",
    "british": "Europe/London", "japan": "Asia/Tokyo",
    "new york": "America/New_York", "chicago": "America/Chicago", "los angeles": "America/Los_Angeles",
    "pacific": "America/Los_Angeles", "eastern": "America/New_York", "central": "America/Chicago",
    "mountain": "America/Denver", "london": "Europe/London", "madrid": "Europe/Madrid",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin", "rotterdam": "Europe/Amsterdam",
    "amsterdam": "Europe/Amsterdam", "tokyo": "Asia/Tokyo", "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney", "dubai": "Asia/Dubai", "toronto": "America/Toronto",
    "denver": "America/Denver", "seattle": "America/Los_Angeles", "boston": "America/New_York",
}

DUR_UNITS = {
    "minute": "minutes", "min": "minutes", "hour": "hours", "h": "hours", "hr": "hours",
    "day": "days", "week": "weeks", "month": "months", "year": "years", "calendar day": "days",
    "business day": "business_days", "working day": "business_days",
}


@dataclass
class Span:
    kind: str
    text: str
    value: Any
    start: int
    end: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.kind}:{self.text.strip().lower()}"


_MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_TIME_RE = r"(?<![\d.,])(?P<h>[01]?\d|2[0-3]):(?P<mi>[0-5]\d)(?!\d|\.\d)(?:\s*(?P<ampm>am|pm))?"
_ZONE_RE = r"(?P<zone>utc\s*[+\-−–]\s*\d{1,2}(?::?\d{2})?|utc\b|[A-Z]{2,4}T\b|(?:" + "|".join(re.escape(c) for c in sorted(CITY_ZONES, key=len, reverse=True)) + r")(?:\s+(?:local\s+)?time)?)"

DATE_PATTERNS = [
    # 31 August 2026 / 3 Sep 2027 / 14 September
    re.compile(r"\b(?P<d>[0-3]?\d)(?:st|nd|rd|th)?[ \t]+(?P<mon>" + _MONTH_RE + r")\.?(?:,?[ \t]+(?P<y>(?:19|20)\d{2}))?\b", re.I),
    # March 11, 2026 / Sep 01 / February 28
    re.compile(r"\b(?P<mon>" + _MONTH_RE + r")\.?[ \t]+(?P<d>[0-3]?\d)(?:st|nd|rd|th)?(?:,?[ \t]+(?P<y>(?:19|20)\d{2}))?\b", re.I),
    # 2026-09-01
    re.compile(r"\b(?P<y>(?:19|20)\d{2})-(?P<mon>0?\d|1[0-2])-(?P<d>[0-3]?\d)\b"),
]
TIME_PAT = re.compile(r"\b" + _TIME_RE + r"\b", re.I)
ZONE_PAT = re.compile(_ZONE_RE, re.I)
DUR_PAT = re.compile(
    r"\b(?P<n>\d+(?:[.,]\d+)?)\s*(?P<u>business\s+days?|working\s+days?|calendar\s+days?|minutes?|mins?|hours?|hrs?|h\b|days?|weeks?|months?|years?)\b", re.I)
PCT_PAT = re.compile(r"(?P<n>\d+(?:[.,]\d+)?)\s*(?:%|percent\b|per\s+cent\b)", re.I)
AMT_PAT = re.compile(
    r"(?:(?P<cur>USD|EUR|GBP|JPY|CHF|CAD|AUD|\$|€|£)\s?)?(?P<n>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s?(?P<cur2>USD|EUR|GBP|JPY|kg|g|lb|lbs|GB|TiB|TB|MB|credits))?\b", re.I)
NUM_PAT = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w])")


def _parse_zone(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip().lower().replace("−", "-").replace("–", "-")
    m = re.match(r"utc\s*([+\-])\s*(\d{1,2})(?::?(\d{2}))?", t)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        off = int(m.group(2)) + (int(m.group(3) or 0) / 60)
        return {"offset_h": sign * off}
    for city, tzname in sorted(CITY_ZONES.items(), key=lambda kv: -len(kv[0])):
        if t.startswith(city):
            return {"iana": tzname}
    if t in ZONES:
        return {"offset_h": ZONES[t]}
    return None


def _year_default(text: str) -> Optional[int]:
    ys = re.findall(r"\b(20\d{2})\b", text)
    return int(max(set(ys), key=ys.count)) if ys else None


def propose(state: str) -> List[Span]:
    spans: List[Span] = []
    default_year = _year_default(state)

    # datetimes: a date, plus an adjacent time and zone if present within ~40 chars.
    for pat in DATE_PATTERNS:
        for m in pat.finditer(state):
            try:
                mon = m.group("mon")
                month = int(mon) if mon.isdigit() else MONTHS[mon.lower()[:3] if mon.lower() not in MONTHS else mon.lower()]
                day = int(m.group("d"))
                y = m.group("y")
                year = int(y) if y else (default_year or datetime.now().year)
                dt = datetime(year, month, day)
            except (ValueError, KeyError):
                continue
            start, end = m.start(), m.end()
            window_after = state[end:end + 45]
            window_before = state[max(0, start - 45):start]
            tm = TIME_PAT.search(window_after) or TIME_PAT.search(window_before)
            zone = None
            if tm:
                h, mi = int(tm.group("h")), int(tm.group("mi"))
                if tm.group("ampm"):
                    if tm.group("ampm").lower() == "pm" and h < 12:
                        h += 12
                    if tm.group("ampm").lower() == "am" and h == 12:
                        h = 0
                dt = dt.replace(hour=h, minute=mi)
                if tm.re.pattern and tm.string is window_after:
                    end = end + tm.end()
                else:
                    start = max(0, start - 45) + tm.start()
            zm = ZONE_PAT.search(state[end:end + 40])
            if zm:
                zone = _parse_zone(zm.group("zone"))
                if zone:
                    end = end + zm.end()
            text = state[start:end].strip(" ,;")
            spans.append(Span("datetime", text, dt, start, end, {"zone": zone, "has_time": bool(tm), "has_year": bool(y)}))

    for m in DUR_PAT.finditer(state):
        n = float(m.group("n").replace(",", "."))
        u = m.group("u").lower().rstrip("s")
        u = "business day" if u.startswith(("business", "working")) else ("day" if u.startswith("calendar") else u)
        unit = DUR_UNITS.get(u, DUR_UNITS.get(u.rstrip("s"), None))
        if unit is None:
            continue
        spans.append(Span("duration", m.group(0), {"n": n, "unit": unit}, m.start(), m.end()))

    for m in ZONE_PAT.finditer(state):
        z = _parse_zone(m.group("zone"))
        if z:
            spans.append(Span("timezone", m.group(0), z, m.start(), m.end()))

    for m in PCT_PAT.finditer(state):
        spans.append(Span("percent", m.group(0), float(m.group("n").replace(",", ".")), m.start(), m.end()))

    taken = [(s.start, s.end) for s in spans]  # datetime/duration/zone/percent already claimed
    for m in AMT_PAT.finditer(state):
        raw = m.group("n")
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        cur = (m.group("cur") or m.group("cur2") or "").upper()
        if any(a < m.end() and m.start() < b for a, b in taken):
            continue
        bare = not cur and "," not in raw and "." not in raw
        # Bare short integers are ordinals, day numbers, clause numbers; bare
        # long ones are ids/serials. Real amounts carry a unit, currency,
        # thousands separator or decimals.
        if bare and (len(raw) <= 2 or len(raw) >= 5 or re.fullmatch(r"(19|20)\d{2}", raw)):
            continue
        rich = bool(cur) or "." in raw or "," in raw
        spans.append(Span("amount", m.group(0).strip(), val, m.start(), m.end(), {"unit": cur, "rich": rich}))

    # Dedupe by (kind, text), keep first occurrence.
    seen = set()
    out: List[Span] = []
    for s in sorted(spans, key=lambda s: (s.start, -s.end)):
        k = s.key()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def by_kind(spans: List[Span], kind: str) -> List[Span]:
    if kind == "number":
        return [s for s in spans if s.kind in ("amount", "percent")]
    return [s for s in spans if s.kind == kind]
