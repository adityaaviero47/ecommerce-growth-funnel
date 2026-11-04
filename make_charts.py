"""Charts for README from results/results.json."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
f = R["funnel"]
stages = ["Sessions", "Viewed a\nproduct", "Reached a\nvalue page", "Purchased"]
vals = [f["sessions"], f["viewed_product"], f["reached_value_page"], f["purchased"]]
ax[0, 0].bar(stages, vals, color=["#9AA5AD", "#9AA5AD", "#E07A2F", "#0052CC"])
for i, v in enumerate(vals): ax[0, 0].text(i, v + 150, f"{v:,}", ha="center", fontsize=9)
ax[0, 0].set_title("Conversion funnel: the big drop is before the value page"); ax[0, 0].set_ylabel("Sessions")

t = R["traffic"]["top5"]
ax[0, 1].bar([f"Source {x['TrafficType']}" for x in t], [x["conversion"] * 100 for x in t],
             color=["#0052CC" if x["TrafficType"] not in R["traffic"]["low_converting_top_sources"] else "#E07A2F" for x in t])
ax[0, 1].axhline(f["conversion"] * 100, color="#555", ls="--", lw=0.8); ax[0, 1].text(3.6, f["conversion"] * 100 + 0.6, "site average", fontsize=8, color="#555")
for i, x in enumerate(t): ax[0, 1].text(i, x["conversion"] * 100 + 0.5, f"{x['share']*100:.0f}% of\nsessions", ha="center", fontsize=8)
ax[0, 1].set_ylim(0, 30); ax[0, 1].set_ylabel("Conversion (%)"); ax[0, 1].set_title("Biggest traffic sources: three convert below average")

g = R["targeting"]
for k, c, lab in [("with_page_values", "#0052CC", "All signals"), ("behaviour_only", "#E07A2F", "Early-session signals")]:
    xs = [0, 10, 20, 30]; ys = [0, g[k]["top10_capture"] * 100, g[k]["top20_capture"] * 100, g[k]["top30_capture"] * 100]
    ax[1, 0].plot(xs, ys, "o-", color=c, label=f"{lab} (AUC {g[k]['auc']:.2f})")
ax[1, 0].plot([0, 30], [0, 30], "--", color="#9AA5AD", label="Random targeting")
ax[1, 0].set_xlabel("Share of sessions targeted (%)"); ax[1, 0].set_ylabel("Share of all buyers reached (%)")
ax[1, 0].set_title("AI targeting: propensity model lift"); ax[1, 0].legend(fontsize=8)

r = R["retention"]
ax[1, 1].bar(["On time", "Late"], [r["one_star_share_on_time"] * 100, r["one_star_share_late"] * 100], color=["#0052CC", "#E07A2F"])
for i, v in enumerate([r["one_star_share_on_time"] * 100, r["one_star_share_late"] * 100]): ax[1, 1].text(i, v + 1, f"{v:.0f}%", ha="center")
ax[1, 1].set_ylabel("Orders rated 1 star (%)"); ax[1, 1].set_title(f"Retention: late deliveries draw 8x the 1-star reviews")
for a in ax.flat: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
