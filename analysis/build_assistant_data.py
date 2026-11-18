"""Builds the per-team data the Recognition Desk demo uses (SYNTHETIC). Output: ../site/assistant/data.js"""
import json
from datetime import date
from pathlib import Path
import pandas as pd

D = Path(__file__).parent / "data"
ASOF = pd.Timestamp("2026-11-15")
emp = pd.read_csv(D / "employees.csv", parse_dates=["join_date"])
rec = pd.read_csv(D / "recognitions.csv", parse_dates=["date"])
names = emp.set_index("employee_id").name.to_dict()
human = rec[rec.giver_id != "SYSTEM"]
mgrs = emp[emp.level == "Manager"].set_index("team_id")

def next_annual(md_or_date, asof):
    """Days until the next occurrence of a month/day."""
    m, d = md_or_date
    for yr in (asof.year, asof.year + 1):
        try:
            t = pd.Timestamp(year=yr, month=m, day=d)
        except ValueError:
            t = pd.Timestamp(year=yr, month=m, day=28)
        if t >= asof.normalize():
            return (t - asof.normalize()).days
    return 999

teams = []
for tid, g in emp.groupby("team_id"):
    mgr = mgrs.loc[tid]
    members = []
    for _, p in g[g.level != "Manager"].iterrows():
        got = human[human.receiver_id == p.employee_id].sort_values("date")
        last = got.iloc[-1] if len(got) else None
        recent = got[got.date > ASOF - pd.Timedelta(days=90)]
        top_val = recent.company_value.value_counts().index[0] if len(recent) else None
        mgr_sent = got[got.giver_id == mgr.employee_id]
        ann = next_annual((p.join_date.month, p.join_date.day), ASOF)
        years = (ASOF + pd.Timedelta(days=ann)).year - p.join_date.year
        bm, bd = map(int, p.birthday.split("-"))
        members.append({
            "id": p.employee_id, "name": p["name"], "level": p.level, "dept": p.department,
            "days_since": int((ASOF - last.date).days) if last is not None else None,
            "received_90d": int(len(recent)),
            "top_value": top_val,
            "last_from": names.get(last.giver_id) if last is not None else None,
            "last_value": last.company_value if last is not None else None,
            "from_mgr_90d": int((mgr_sent.date > ASOF - pd.Timedelta(days=90)).sum()),
            "anniv_in": ann if ann <= 21 and years >= 1 else None,
            "years": int(years),
            "bday_in": next_annual((bm, bd), ASOF) if next_annual((bm, bd), ASOF) <= 7 else None,
        })
    teams.append({"team": tid, "dept": g.department.iloc[0], "manager": mgr["name"], "manager_id": mgr.employee_id,
                  "members": members})
teams.sort(key=lambda t: t["team"])
out = {"asof": str(ASOF.date()), "teams": teams}
Path(__file__).parent.parent.joinpath("assistant/data.js").write_text("window.DESK_DATA=" + json.dumps(out, separators=(",", ":")) + ";")
n = sum(len(t["members"]) for t in teams)
due = sum(1 for t in teams for m in t["members"] if m["days_since"] is None or m["days_since"] >= 30)
print(len(teams), "teams", n, "members;", due, "silent 30d+;", sum(1 for t in teams for m in t["members"] if m["anniv_in"] is not None), "anniversaries ≤21d")
