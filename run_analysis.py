"""E-commerce growth funnel: where customers drop off, which levers move conversion and retention,
and how an ML propensity model can target campaigns.

Part A, acquisition and conversion: 12,330 sessions from the UCI Online Shoppers Purchasing
Intention dataset (one year of sessions on a real online store).
Part B, retention: 99,441 orders from the Olist Brazilian e-commerce public dataset.
Reproduces every number in README.md:  python run_analysis.py"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

R = {}
# ================= Part A: acquisition and conversion =================
s = pd.read_csv("data/online_shoppers_intention.csv.gz")
s["buy"] = s.Revenue.astype(bool)
n = len(s); conv = s.buy.mean()
# funnel: session -> viewed a product page -> reached a page that leads to purchases (PageValues > 0) -> purchase
st1 = s.ProductRelated > 0
st2 = st1 & (s.PageValues > 0)
R["funnel"] = dict(sessions=n, viewed_product=int(st1.sum()), reached_value_page=int(st2.sum()), purchased=int(s.buy.sum()),
                   conversion=float(conv),
                   product_to_value_page_drop=float(1 - st2.sum() / st1.sum()),
                   value_page_to_purchase=float(s.buy[st2].mean()), purchase_rate_without_value_page=float(s.buy[~st2].mean()))
vt = s.groupby("VisitorType").agg(sessions=("buy", "size"), conversion=("buy", "mean"))
vt["share"] = vt.sessions / n
R["visitor_type"] = vt.round(4).reset_index().to_dict("records")
tt = s.groupby("TrafficType").agg(sessions=("buy", "size"), buyers=("buy", "sum"), conversion=("buy", "mean")).sort_values("sessions", ascending=False)
tt["share"] = tt.sessions / n
top = tt.head(5).copy()
# growth lever: lift each of the 5 biggest sources with below-average conversion up to the site average
low = top[top.conversion < conv - 0.02]          # clearly below average (more than 2 points)
extra = float(((conv - low.conversion) * low.sessions).sum())
R["traffic"] = dict(top5=top.round(4).reset_index().to_dict("records"), low_converting_top_sources=low.index.tolist(),
                    low_share_of_sessions=float(low.sessions.sum() / n), extra_purchases_if_at_average=extra,
                    uplift_in_total_purchases=extra / float(s.buy.sum()))
mo = s.groupby("Month").agg(sessions=("buy", "size"), conversion=("buy", "mean"))
R["month"] = dict(best=mo.conversion.idxmax(), best_conv=float(mo.conversion.max()), worst=mo.conversion.idxmin(),
                  worst_conv=float(mo.conversion.min()), nov_share_of_purchases=float(s.buy[s.Month == "Nov"].sum() / s.buy.sum()))
R["bounce"] = dict(conv_high_bounce=float(s.buy[s.BounceRates >= 0.05].mean()), share_high_bounce=float((s.BounceRates >= 0.05).mean()),
                   conv_low_bounce=float(s.buy[s.BounceRates < 0.05].mean()))

# AI targeting: purchase-propensity model; one version without PageValues (usable early in a session)
X = s.drop(columns=["Revenue", "buy"]).copy()
for c in ["Month", "VisitorType", "Weekend"]:
    X[c] = X[c].astype("category").cat.codes
cat_cols = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
y = s.buy.astype(int).to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=7)
def model(cols):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          categorical_features=[c in cat_cols for c in cols], random_state=7)
def lift(p, frac):
    k = int(len(p) * frac); idx = np.argsort(-p)[:k]; return float(y[idx].sum() / y.sum())
out = {}
for name, cols in [("with_page_values", list(X.columns)), ("behaviour_only", [c for c in X.columns if c != "PageValues"])]:
    p = cross_val_predict(model(cols), X[cols], y, cv=cv, method="predict_proba")[:, 1]
    out[name] = dict(auc=float(roc_auc_score(y, p)), top10_capture=lift(p, 0.10), top20_capture=lift(p, 0.20),
                     top30_capture=lift(p, 0.30), top20_conversion=float(y[np.argsort(-p)[: int(n * 0.2)]].mean()))
R["targeting"] = out
R["targeting"]["note"] = "5-fold cross-validated; capture = share of all purchases inside the top-scored share of sessions"

# ================= Part B: retention =================
o = pd.read_csv("data/olist_orders_dataset.csv.gz", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
c = pd.read_csv("data/olist_customers_dataset.csv.gz")
rv = pd.read_csv("data/olist_order_reviews_dataset.csv.gz").groupby("order_id").review_score.mean()
pay = pd.read_csv("data/olist_order_payments_dataset.csv.gz").groupby("order_id").payment_value.sum()
o = o.merge(c[["customer_id", "customer_unique_id"]], on="customer_id")
o = o[o.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"]).copy()
o["late"] = o.order_delivered_customer_date.dt.normalize() > o.order_estimated_delivery_date
o["days"] = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.days
o["review"] = o.order_id.map(rv); o["value"] = o.order_id.map(pay)
o = o.sort_values("order_purchase_timestamp")
o["n"] = o.groupby("customer_unique_id").cumcount() + 1
first = o[o.n == 1].copy()
nord = o.groupby("customer_unique_id").size()
first["repeat"] = first.customer_unique_id.map(nord) > 1
second = o[o.n == 2].set_index("customer_unique_id").order_purchase_timestamp
first["gap"] = (first.customer_unique_id.map(second) - first.order_purchase_timestamp).dt.days
# customers who bought early enough to have had 6 months to come back
cutoff = o.order_purchase_timestamp.max() - pd.Timedelta(days=183)
f6 = first[first.order_purchase_timestamp <= cutoff]
rep = lambda d: float(d.repeat.mean())
R["retention"] = dict(
    delivered_orders=int(len(o)), customers=int(len(first)), repeat_rate=rep(first), repeat_rate_6m_window=rep(f6),
    median_days_to_second_order=float(first.gap.median()),
    late_share=float(o.late.mean()), review_on_time=float(o.review[~o.late].mean()), review_late=float(o.review[o.late].mean()),
    one_star_share_on_time=float((o.review[~o.late] <= 1).mean()), one_star_share_late=float((o.review[o.late] <= 1).mean()),
    repeat_after_good_review=rep(first[first.review >= 4]), repeat_after_bad_review=rep(first[first.review <= 2]),
    repeat_first_on_time=rep(first[~first.late]), repeat_first_late=rep(first[first.late]),
    avg_order_value=float(o.value.mean()), median_delivery_days=float(o.days.median()))
# business cases. (1) one more point of repeat rate: 1% of customers place one extra order at the average value.
# (2) late deliveries: share of 1-star reviews and repeat rate, late vs on time (association, not proven cause)
r_ = R["retention"]; total_rev = float(o.value.sum())
extra_orders = 0.01 * len(first)
R["retention"]["business_case"] = dict(
    extra_orders_per_point=float(extra_orders), extra_revenue_brl=float(extra_orders * r_["avg_order_value"]),
    share_of_revenue=float(extra_orders * r_["avg_order_value"] / total_rev),
    repeat_gap_late_vs_on_time=float(1 - r_["repeat_first_late"] / r_["repeat_first_on_time"]),
    one_star_ratio_late_vs_on_time=float(r_["one_star_share_late"] / r_["one_star_share_on_time"]))
json.dump(R, open("results/results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=2, default=float))
