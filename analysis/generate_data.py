"""
Synthetic employee-recognition dataset for a portfolio project.

EVERYTHING HERE IS SYNTHETIC. No real company, employee or customer data is used.
The shape (HRMS-synced employees, peer and manager recognition, points, redemptions,
a rollout in waves) is modelled on how recognition platforms are publicly described.

A manager-nudge A/B test is PLANTED in the data (from 2026-09-07) so that the dashboard
can demonstrate the experiment-analysis method. The effect size is chosen by this script,
so the result shows how to analyse a test, not what a real test would find.

Run:  python3 generate_data.py        (deterministic, seed 2026)
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)

LAUNCH = date(2026, 6, 1)          # a Monday
N_WEEKS = 24                       # 1 Jun to 15 Nov 2026
END = LAUNCH + timedelta(weeks=N_WEEKS) - timedelta(days=1)
AB_START_WEEK = 14                 # 7 Sep 2026
AI_START_WEEK = 8                  # AI-assisted drafting available from 27 Jul 2026
NUDGE_LIFT = 0.15                  # planted: cuts a responsive manager's "no recognition this week" chance by 15%
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# name: (headcount, engagement multiplier, rollout wave (weeks after launch), login rate)
DEPTS = {
    "Engineering": (150, 0.95, 1, 0.80),
    "Sales": (100, 1.35, 0, 0.90),
    "Support": (80, 0.70, 3, 0.66),
    "Operations": (60, 0.85, 3, 0.74),
    "Marketing": (50, 1.20, 0, 0.88),
    "Product": (40, 1.10, 1, 0.86),
    "Finance": (35, 0.80, 3, 0.72),
    "Design": (30, 1.10, 1, 0.87),
    "HR": (25, 1.30, 0, 0.94),
    "IT and Legal": (30, 0.75, 3, 0.70),
}
FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Ishaan", "Kabir", "Rahul", "Siddharth",
         "Ananya", "Diya", "Isha", "Kavya", "Meera", "Neha", "Priya", "Riya", "Sneha", "Tanvi",
         "Aisha", "Divya", "Pooja", "Shreya", "Nikhil", "Varun", "Manish", "Deepak", "Lakshmi", "Harini",
         "Suresh", "Ramesh", "Anjali", "Pallavi", "Gaurav", "Mohit", "Swati", "Ritika", "Vikram", "Yash"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Mehta", "Gupta", "Singh", "Kumar",
        "Das", "Banerjee", "Menon", "Pillai", "Joshi", "Kulkarni", "Deshmukh", "Rao", "Shetty", "Bhat",
        "Chopra", "Malhotra", "Kapoor", "Khanna", "Agarwal", "Jain", "Saxena", "Mishra", "Pandey", "Naidu"]
LOCATIONS = ["Bengaluru", "Pune", "Hyderabad", "Delhi NCR", "Chennai"]
LOC_P = [0.38, 0.20, 0.17, 0.15, 0.10]
VALUES = ["Customer first", "Ownership", "Innovation", "Teamwork", "Integrity"]
CHANNELS = ["Slack", "Microsoft Teams", "Web", "Mobile"]
CHAN_P = [0.36, 0.28, 0.20, 0.16]

# ---------------------------------------------------------------- employees and teams
rows, teams = [], []
eid = 0
for dept, (n, mult, wave, login) in DEPTS.items():
    n_teams = max(2, round(n / 9))
    sizes = [n // n_teams + (1 if i < n % n_teams else 0) for i in range(n_teams)]
    for t, size in enumerate(sizes, start=1):
        team_id = f"{dept.replace(' and ', '').replace(' ', '')[:3].upper()}-{t:02d}"
        mgr_id = None
        for k in range(size):
            eid += 1
            emp = f"E{eid:04d}"
            is_mgr = k == 0
            if is_mgr:
                mgr_id = emp
            join = date(2026, 6, 1) + timedelta(days=int(rng.integers(0, 168))) if rng.random() < 0.12 \
                else date(2018, 1, 1) + timedelta(days=int(rng.integers(0, 3070)))
            rows.append(dict(
                employee_id=emp,
                name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                department=dept, team_id=team_id,
                manager_id="" if is_mgr else mgr_id,
                level="Manager" if is_mgr else ("Senior IC" if rng.random() < 0.3 else "IC"),
                location=rng.choice(LOCATIONS, p=LOC_P),
                join_date=join,
                birthday=f"{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}",
            ))
        teams.append(dict(team_id=team_id, department=dept, manager_id=mgr_id, headcount=size))
emp = pd.DataFrame(rows)
team_df = pd.DataFrame(teams)
N = len(emp)

emp["dept_launch"] = emp["department"].map(lambda d: LAUNCH + timedelta(weeks=DEPTS[d][2]))
emp["invited_date"] = [max(a, b) for a, b in zip(emp["dept_launch"], emp["join_date"])]

# HRMS sync invites everyone; only some log in. Managers log in more often.
login_rate = emp["department"].map(lambda d: DEPTS[d][3]).to_numpy() + np.where(emp["level"] == "Manager", 0.08, 0)
logs_in = rng.random(N) < np.clip(login_rate, 0, 0.97)
lag = rng.geometric(0.33, N)  # 1 to ~10 days
emp["first_login_date"] = [
    (inv + timedelta(days=int(l))) if li and (inv + timedelta(days=int(l))) <= END else None
    for inv, l, li in zip(emp["invited_date"], lag, logs_in)
]

# ---------------------------------------------------------------- behaviour parameters
is_mgr = (emp["level"] == "Manager").to_numpy()
dept_mult = emp["department"].map(lambda d: DEPTS[d][1]).to_numpy()
silent = rng.random(N) < 0.40                                   # 40% never send peer kudos
peer_prop = rng.gamma(0.9, 1.0, N) * ~silent
popularity = rng.gamma(0.7, 1.0, N)                             # skewed: some people receive far more
p_mgr = np.clip(rng.beta(2, 3, N), 0.02, 0.9)
p_mgr = np.where(rng.random(N) < 0.35, p_mgr * 0.25, p_mgr)     # 35% of managers rarely recognise
responsive = rng.random(N) < 0.80                               # who reacts to nudges

# A/B assignment at team level, blocked by department and by the manager's baseline activity
# (teams are sorted by that baseline, paired off, and one team per pair is picked at random).
mgr_idx = {m: i for i, m in enumerate(emp.employee_id)}
assign = []
for dept, grp in team_df.groupby("department"):
    grp = grp.assign(base=[p_mgr[mgr_idx[m]] for m in grp.manager_id]).sort_values("base")
    rows_ = list(grp.itertuples())
    for k in range(0, len(rows_), 2):
        block = rows_[k:k + 2]
        flip = rng.random() < 0.5
        for j, r in enumerate(block):
            assign.append(dict(team_id=r.team_id, manager_id=r.manager_id, department=dept,
                               variant="nudge" if (j == 0) == flip else "control",
                               assignment_date=LAUNCH + timedelta(weeks=AB_START_WEEK)))
ab = pd.DataFrame(assign)
variant_of_team = dict(zip(ab.team_id, ab.variant))

team_members = emp.groupby("team_id").apply(lambda g: g.index.to_list(), include_groups=False).to_dict()
team_of = emp["team_id"].to_numpy()
invited_np = np.array(emp["invited_date"], dtype="datetime64[D]")
login_np = np.array([np.datetime64(d, "D") if d else np.datetime64("NaT", "D") for d in emp["first_login_date"]], dtype="datetime64[D]")
join_np = np.array(emp["join_date"], dtype="datetime64[D]")

recv_log = [[] for _ in range(N)]
events = []
mgr_weeks = []
weekday_w = np.array([1.1, 1.2, 1.2, 1.2, 1.1, 0.2, 0.1]); weekday_w /= weekday_w.sum()


def activity(w):  # launch spike, then settle below the spike but above zero
    return 0.60 + 1.40 * np.exp(-w / 4.0)


def points_for(kind):
    return {"Peer kudos": int(rng.choice([20, 25, 30, 40, 50])),
            "Manager recognition": int(rng.choice([50, 75, 100])),
            "Spot award": int(rng.choice([100, 150, 200, 250]))}[kind]


def add_event(when, giver, receiver, kind, week, team_variant):
    ai = week >= AI_START_WEEK and rng.random() < 0.35
    words = max(4, int(rng.poisson(17) + (11 if ai else 0) - (1 if team_variant == "nudge" and week >= AB_START_WEEK else 0)))
    events.append(dict(
        date=when, giver_id=giver, receiver_id=receiver, type=kind, points=points_for(kind),
        company_value=rng.choice(VALUES), channel=rng.choice(CHANNELS, p=CHAN_P),
        ai_assisted=ai, message_words=words,
    ))
    recv_log[emp.index[emp.employee_id == receiver][0]].append(when)


def trailing_recv(i, when):
    return sum(1 for d in recv_log[i] if 0 <= (when - d).days <= 30)


def pick_peer_receiver(s, when, least=False):
    d64 = np.datetime64(when)
    if rng.random() < 0.75 or least:
        pool = [i for i in team_members[team_of[s]] if i != s and invited_np[i] <= d64]
        if pool:
            if least:
                pool = sorted(pool, key=lambda i: (trailing_recv(i, when), rng.random()))[:3]
                return int(rng.choice(pool))
            wts = popularity[pool]
            return int(rng.choice(pool, p=wts / wts.sum()))
    mask = (invited_np <= d64) & (np.arange(N) != s)
    pool = np.flatnonzero(mask)
    wts = popularity[pool]
    return int(rng.choice(pool, p=wts / wts.sum()))


# ---------------------------------------------------------------- weekly simulation
for w in range(N_WEEKS):
    ws = LAUNCH + timedelta(weeks=w)
    we = ws + timedelta(days=6)
    act = activity(w)
    for s in range(N):
        if np.isnat(login_np[s]) or login_np[s] > np.datetime64(we):
            continue
        in_post = w >= AB_START_WEEK
        v = variant_of_team[team_of[s]]
        # peer kudos from everyone
        lam = 0.65 * peer_prop[s] * dept_mult[s] * act * (1.08 if (in_post and v == "nudge") else 1.0)
        for _ in range(rng.poisson(lam)):
            when = ws + timedelta(days=int(rng.choice(7, p=weekday_w)))
            if when < login_np[s].astype(object):
                continue
            r = pick_peer_receiver(s, when)
            add_event(when, emp.employee_id[s], emp.employee_id[r], "Peer kudos", w, v)
        # manager recognition
        if is_mgr[s]:
            p = float(np.clip(p_mgr[s] * (0.5 + 0.5 * act / activity(0)), 0, 0.95))
            if in_post and v == "nudge" and responsive[s]:
                p = 1 - (1 - p) * (1 - NUDGE_LIFT)
            sent = rng.random() < p
            n_send = 1 + int(rng.poisson(0.3)) if sent else 0
            for _ in range(n_send):
                when = ws + timedelta(days=int(rng.choice(5)))
                least = in_post and v == "nudge"
                r = pick_peer_receiver(s, when, least=least) if team_members[team_of[s]].__len__() > 1 else s
                kind = "Spot award" if rng.random() < 0.3 else "Manager recognition"
                add_event(when, emp.employee_id[s], emp.employee_id[r], kind, w, v)
            mgr_weeks.append(dict(manager_id=emp.employee_id[s], team_id=team_of[s], week_start=ws,
                                  week_index=w, sent=int(n_send > 0), recognitions_sent=n_send,
                                  variant=v, period="post" if in_post else "pre"))

# automatic milestones from HRMS data (birthdays and work anniversaries), system-generated
for i in range(N):
    for d in pd.date_range(max(emp.invited_date[i], LAUNCH), END):
        d0 = d.date()
        if f"{d0.month:02d}-{d0.day:02d}" == emp.birthday[i]:
            events.append(dict(date=d0, giver_id="SYSTEM", receiver_id=emp.employee_id[i], type="Milestone (auto)",
                               points=50, company_value="Teamwork", channel="Email", ai_assisted=False, message_words=0))
            recv_log[i].append(d0)
        jd = emp.join_date[i]
        if d0.year > jd.year and (d0.month, d0.day) == (jd.month, jd.day):
            yrs = d0.year - jd.year
            events.append(dict(date=d0, giver_id="SYSTEM", receiver_id=emp.employee_id[i], type="Milestone (auto)",
                               points=100 * min(yrs, 5), company_value="Ownership", channel="Email", ai_assisted=False, message_words=0))
            recv_log[i].append(d0)

rec = pd.DataFrame(events).sort_values(["date", "receiver_id"]).reset_index(drop=True)
rec.insert(0, "recognition_id", [f"R{i + 1:05d}" for i in range(len(rec))])

# ---------------------------------------------------------------- redemptions
CATS = ["Gift card", "Experience", "Merchandise", "Charity donation", "Wellness"]
CAT_P = [0.55, 0.15, 0.12, 0.08, 0.10]
redeem, balance = [], np.zeros(N)
by_week = {w: g for w, g in rec.groupby(((pd.to_datetime(rec["date"]) - pd.Timestamp(LAUNCH)).dt.days // 7))}
idx_of = {e: i for i, e in enumerate(emp.employee_id)}
for w in range(N_WEEKS):
    if w in by_week:
        for r_id, pts in by_week[w].groupby("receiver_id")["points"].sum().items():
            balance[idx_of[r_id]] += pts
    ws = LAUNCH + timedelta(weeks=w)
    for i in range(N):
        if np.isnat(login_np[i]) or login_np[i] > np.datetime64(ws + timedelta(days=6)) or balance[i] < 300:
            continue
        if rng.random() < (0.18 if balance[i] >= 500 else 0.12):
            amt = int(rng.choice([a for a in (300, 500, 750, 1000) if a <= balance[i]]))
            balance[i] -= amt
            cat = rng.choice(CATS, p=CAT_P)
            instant = (cat in ("Gift card", "Charity donation")) and rng.random() < 0.75
            days = 0 if instant else int(np.clip(rng.normal({"Experience": 8, "Merchandise": 7, "Wellness": 4}.get(cat, 6), 2), 2, 14))
            redeem.append(dict(date=ws + timedelta(days=int(rng.integers(0, 7))), employee_id=emp.employee_id[i],
                               points=amt, reward_category=cat, delivery_type="Instant" if instant else "Standard",
                               delivery_days=days))
red = pd.DataFrame(redeem).sort_values("date").reset_index(drop=True)
red.insert(0, "redemption_id", [f"X{i + 1:05d}" for i in range(len(red))])

# ---------------------------------------------------------------- funnel columns and save
first_sent = rec[rec.giver_id != "SYSTEM"].groupby("giver_id")["date"].min()
first_recv = rec[rec.giver_id != "SYSTEM"].groupby("receiver_id")["date"].min()
first_redeem = red.groupby("employee_id")["date"].min()
emp["first_recognition_sent_date"] = emp.employee_id.map(first_sent)
emp["first_recognition_received_date"] = emp.employee_id.map(first_recv)
emp["first_redemption_date"] = emp.employee_id.map(first_redeem)
emp = emp.drop(columns=["dept_launch"])

emp.to_csv(OUT / "employees.csv", index=False)
team_df.to_csv(OUT / "teams.csv", index=False)
ab.to_csv(OUT / "experiment_assignments.csv", index=False)
rec.to_csv(OUT / "recognitions.csv", index=False)
red.to_csv(OUT / "redemptions.csv", index=False)
pd.DataFrame(mgr_weeks).to_csv(OUT / "manager_weeks.csv", index=False)

print(f"employees={len(emp)} teams={len(team_df)} recognitions={len(rec)} redemptions={len(red)} manager_weeks={len(mgr_weeks)}")
