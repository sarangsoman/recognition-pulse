"""
Turns the synthetic CSVs into the aggregated numbers the dashboard shows, and runs the A/B analysis.
Output: data/dashboard_data.json   (SYNTHETIC DATA, see generate_data.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path(__file__).parent / "data"
END = pd.Timestamp("2026-11-15")
LAUNCH = pd.Timestamp("2026-06-01")
AB_START = pd.Timestamp("2026-09-07")

emp = pd.read_csv(D / "employees.csv", parse_dates=["join_date", "invited_date", "first_login_date",
                  "first_recognition_sent_date", "first_recognition_received_date", "first_redemption_date"])
rec = pd.read_csv(D / "recognitions.csv", parse_dates=["date"])
red = pd.read_csv(D / "redemptions.csv", parse_dates=["date"])
mw = pd.read_csv(D / "manager_weeks.csv")
teams = pd.read_csv(D / "teams.csv")
ab = pd.read_csv(D / "experiment_assignments.csv")
human = rec[rec.giver_id != "SYSTEM"]


def window(df, end, days=30):
    return df[(df.date > end - pd.Timedelta(days=days)) & (df.date <= end)]


def participation(end):
    w = window(human, end)
    active = set(w.giver_id) | set(w.receiver_id)
    invited = emp[emp.invited_date <= end]
    return len(active & set(invited.employee_id)) / len(invited)


out = {"asof": END.strftime("%Y-%m-%d")}

# ---- headline numbers (last 30 days)
w30 = window(human, END)
mgrs = emp[emp.level == "Manager"]
mgr_sent_30 = set(w30[w30.giver_id.isin(mgrs.employee_id)].giver_id)
logged = emp[emp.first_login_date.notna()]
recv_30 = set(window(rec, END).receiver_id)
out["kpis"] = {
    "employees": int(len(emp)),
    "login_rate": round(len(logged) / len(emp), 3),
    "participation_30d": round(participation(END), 3),
    "manager_coverage_30d": round(len(mgr_sent_30) / len(mgrs), 3),
    "recognised_share_30d": round(len(recv_30) / len(emp), 3),
    "recognitions_total": int(len(rec)),
    "human_recognitions_30d": int(len(w30)),
    "redeemers_share": round(red.employee_id.nunique() / len(logged), 3),
    "instant_share": round((red.delivery_type == "Instant").mean(), 3),
    "avg_standard_delivery_days": round(red[red.delivery_type == "Standard"].delivery_days.mean(), 1),
}

# ---- weekly series
weeks = pd.date_range(LAUNCH, END, freq="7D")
series = []
for ws in weeks:
    we = ws + pd.Timedelta(days=6)
    wk = human[(human.date >= ws) & (human.date <= we)]
    invited = (emp.invited_date <= we).sum()
    active = set(wk.giver_id) | set(wk.receiver_id)
    series.append({
        "week": ws.strftime("%Y-%m-%d"),
        "recognitions": int(len(wk)),
        "active_share": round(len(active) / invited, 3),
        "ai_share": round(float(wk.ai_assisted.mean()), 3) if len(wk) else 0,
    })
out["weekly"] = series

# ---- funnel
n_inv = len(emp)
n_login = emp.first_login_date.notna().sum()
n_sent = emp.first_recognition_sent_date.notna().sum()
n_sent_7d = ((emp.first_recognition_sent_date - emp.first_login_date).dt.days <= 7).sum()
n_red = emp.first_redemption_date.notna().sum()
out["funnel"] = [
    {"stage": "Invited via HRMS sync", "count": int(n_inv)},
    {"stage": "Logged in", "count": int(n_login)},
    {"stage": "Sent a first recognition", "count": int(n_sent)},
    {"stage": "Redeemed points", "count": int(n_red)},
]
out["funnel_extra"] = {
    "sent_within_7d_of_login": int(n_sent_7d),
    "median_days_login_to_first_send": float((emp.first_recognition_sent_date - emp.first_login_date).dt.days.median()),
}

# ---- departments
dep = []
for d, g in emp.groupby("department"):
    ids = set(g.employee_id)
    wk = window(human, END)
    active = (set(wk.giver_id) | set(wk.receiver_id)) & ids
    gm = g[g.level == "Manager"]
    ms = mw[(mw.manager_id.isin(gm.employee_id)) & (mw.week_index >= 18)]
    dep.append({
        "department": d, "headcount": int(len(g)),
        "login_rate": round(g.first_login_date.notna().mean(), 3),
        "participation_30d": round(len(active) / len(g), 3),
        "manager_weekly_rate": round(float(ms.sent.mean()), 3) if len(ms) else None,
        "sent_per_head_month": round(len(human[human.giver_id.isin(ids) & (human.date > END - pd.Timedelta(days=30))]) / len(g), 2),
    })
out["departments"] = sorted(dep, key=lambda r: r["participation_30d"])

# ---- team coverage (who received something in the last 30 days) and quiet teams
w30_all = window(rec, END)
recv_ids = set(w30_all.receiver_id)
trow = []
for t, g in emp.groupby("team_id"):
    m = g[g.level == "Manager"].employee_id.iloc[0]
    trow.append({
        "team": t, "department": g.department.iloc[0], "headcount": int(len(g)),
        "recognised_30d": round(g.employee_id.isin(recv_ids).mean(), 3),
        "manager_sent_30d": bool(m in mgr_sent_30),
        "variant": ab.set_index("team_id").variant[t],
    })
trow.sort(key=lambda r: (r["recognised_30d"], r["manager_sent_30d"]))
out["quiet_teams"] = trow[:10]

# ---- inclusion: who receives recognition
recv_counts = human.receiver_id.value_counts().reindex(emp.employee_id, fill_value=0).to_numpy()
srt = np.sort(recv_counts)
n = len(srt)
gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum()))
top10 = float(np.sort(recv_counts)[::-1][: n // 10].sum() / recv_counts.sum())
never = float(((recv_counts == 0) & emp.first_login_date.notna().to_numpy()).sum() / emp.first_login_date.notna().sum())
out["inclusion"] = {"gini": round(gini, 2), "top10_share": round(top10, 3), "never_recognised_share": round(never, 3)}

# ---- rewards
cat = red.groupby("reward_category").agg(redemptions=("points", "size"), avg_delivery_days=("delivery_days", "mean"),
                                         instant=("delivery_type", lambda s: (s == "Instant").mean())).round(2)
out["rewards"] = [dict(category=i, **r) for i, r in cat.reset_index().set_index("reward_category").iterrows()]

# ---- A/B test: does a weekly manager nudge raise manager recognition?
by = mw.groupby(["manager_id", "variant", "period"]).sent.mean().unstack("period").reset_index()
by = by.dropna(subset=["pre", "post"])
by["nudge"] = (by.variant == "nudge").astype(float)
X = np.column_stack([np.ones(len(by)), by.nudge, by.pre])
y = by.post.to_numpy()
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - X.shape[1]
cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
se = float(np.sqrt(cov[1, 1]))
tcrit = stats.t.ppf(0.975, dof)
p_ancova = float(2 * (1 - stats.t.cdf(abs(beta[1] / se), dof)))
raw = stats.ttest_ind(by[by.nudge == 1].post, by[by.nudge == 0].post, equal_var=False)
sd_post = float(by.post.std())
n_arm = int(by.groupby("variant").size().min())
mde = float(2.8 * sd_post * np.sqrt(2 / n_arm))  # ~80% power, alpha 0.05, no covariate adjustment
msg = rec[(rec.date >= AB_START) & (rec.giver_id != "SYSTEM")].merge(
    emp[["employee_id", "team_id"]], left_on="giver_id", right_on="employee_id").merge(ab[["team_id", "variant"]], on="team_id")
cov_end = []
for v in ("control", "nudge"):
    ids = emp[emp.team_id.isin(ab[ab.variant == v].team_id)]
    cov_end.append(round(float(ids.employee_id.isin(recv_ids).mean()), 3))
arm = by.groupby("variant")[["pre", "post"]].mean().round(3)
out["ab"] = {
    "teams": {"control": int((ab.variant == "control").sum()), "nudge": int((ab.variant == "nudge").sum())},
    "managers_analysed": int(len(by)),
    "pre": {"control": float(arm.loc["control", "pre"]), "nudge": float(arm.loc["nudge", "pre"])},
    "post": {"control": float(arm.loc["control", "post"]), "nudge": float(arm.loc["nudge", "post"])},
    "did_pp": round(float((arm.loc["nudge", "post"] - arm.loc["nudge", "pre"]) - (arm.loc["control", "post"] - arm.loc["control", "pre"])) * 100, 1),
    "ancova_effect_pp": round(float(beta[1]) * 100, 1),
    "ancova_ci_pp": [round(float((beta[1] - tcrit * se) * 100), 1), round(float((beta[1] + tcrit * se) * 100), 1)],
    "ancova_p": round(p_ancova, 3),
    "raw_post_diff_pp": round(float((by[by.nudge == 1].post.mean() - by[by.nudge == 0].post.mean()) * 100), 1),
    "raw_p": round(float(raw.pvalue), 3),
    "mde_pp": round(mde * 100, 1),
    "team_coverage_end": {"control": cov_end[0], "nudge": cov_end[1]},
    "message_words_post": {v: round(float(msg[msg.variant == v].message_words.mean()), 1) for v in ("control", "nudge")},
}

# ---- AI assistance and channels
ai = human[human.date >= pd.Timestamp("2026-07-27")]
out["ai"] = {
    "share_after_launch": round(float(ai.ai_assisted.mean()), 3),
    "words_ai": round(float(ai[ai.ai_assisted].message_words.mean()), 1),
    "words_manual": round(float(ai[~ai.ai_assisted].message_words.mean()), 1),
}
out["channels"] = human.channel.value_counts(normalize=True).round(3).to_dict()

(D / "dashboard_data.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("kpis", "funnel", "funnel_extra", "inclusion", "ab", "ai")}, indent=1))
print("\nDEPARTMENTS (quietest first)")
for r in out["departments"]:
    print(r)
print("\nQUIET TEAMS", out["quiet_teams"][:4])
