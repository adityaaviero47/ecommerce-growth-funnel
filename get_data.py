"""Download the public datasets and write the trimmed, compressed copies the analysis reads (data/*.csv.gz).
Only needed if data/ is empty:  python get_data.py"""
import os, sys, pandas as pd

SOURCES = {  # public GitHub mirrors of the original datasets
    "online_shoppers_intention.csv": ("https://raw.githubusercontent.com/sidneykung/online_shopper_intentions/main/data/online_shoppers_intention.csv", None),
    "olist_orders_dataset.csv": ("https://raw.githubusercontent.com/Haikalin/Streamlit_Data/main/dataset/orders_dataset.csv",
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]),
    "olist_customers_dataset.csv": ("https://raw.githubusercontent.com/xor9237/brazil_ecommerce/master/olist_customers_dataset.csv",
        ["customer_id", "customer_unique_id"]),
    "olist_order_reviews_dataset.csv": ("https://raw.githubusercontent.com/xor9237/brazil_ecommerce/master/olist_order_reviews_dataset.csv",
        ["order_id", "review_score"]),
    "olist_order_payments_dataset.csv": ("https://raw.githubusercontent.com/xor9237/brazil_ecommerce/master/olist_order_payments_dataset.csv",
        ["order_id", "payment_value"]),
}

def build(locate=lambda name, url: url):
    os.makedirs("data", exist_ok=True)
    for name, (url, cols) in SOURCES.items():
        d = pd.read_csv(locate(name, url), usecols=cols) if cols else pd.read_csv(locate(name, url))
        d.to_csv(f"data/{name}.gz", index=False, compression="gzip")
        print(f"data/{name}.gz  {len(d):,} rows")

if __name__ == "__main__":
    build()
