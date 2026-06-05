import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import norm

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "raw_data"
OUT = BASE_DIR / "data"
OUT.mkdir(exist_ok=True)

#CONFIGURATION
SERVICE_LEVEL_BY_CATEGORY = {
    "DAIRY": 0.98,
    "GROCERY": 0.96,
    "BEVERAGES": 0.95,
    "SNACKS": 0.93,
    "HOMECARE": 0.92,
    "PERSONALCARE": 0.94,
}

#HELPER METHODS
def column_conversion_to_lower(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def column_conversion_to_upper(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.upper()
    return df

def fill_median(df, col, group_cols=None):
    if group_cols:
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())
    return df

# Data Loading
def load_all():
    stores = pd.read_csv(RAW / "stores.csv")
    sales = pd.read_csv(RAW / "sales_daily.csv")
    inv = pd.read_csv(RAW / "inventory_daily.csv")
    po = pd.read_csv(RAW / "purchase_orders.csv")

    with open(RAW / "products.json", "r", encoding="utf-8") as f:
        products = pd.DataFrame(json.load(f))

    return stores, sales, inv, po, products

# Data Cleaning
def clean_inputs(sales, inv, po, products):
    # normalize column names
    sales, inv, po, products = map(column_conversion_to_lower, [sales, inv, po, products])

    # parse dates
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")

    # normalize IDs
    sales = column_conversion_to_upper(sales, ["store_id", "sku_id"])
    inv = column_conversion_to_upper(inv, ["store_id", "sku_id"])
    po = column_conversion_to_upper(po, ["store_id", "sku_id"])
    products = column_conversion_to_upper(products, ["sku_id"])
    products["category"] = products["category"].astype(str).str.strip().str.upper()

    # drop exact duplicates
    sales = sales.drop_duplicates()
    po = po.drop_duplicates()

    key = ["date", "store_id", "sku_id"]
    if sales.duplicated(key).any():
        sales = sales.groupby(key, as_index=False).agg({
            "units_sold": "sum",
            "true_demand_units": "sum",
            "promo_flag": "max",
            "holiday_flag": "max",
            "day_of_week": "first"
        })

    if "on_hand_close" not in inv.columns:
        raise ValueError("inventory_daily.csv must contain 'on_hand_close'")
    inv = inv.rename(columns={"on_hand_close": "on_hand_units"})

    #missing value with median
    inv = fill_median(inv, "on_hand_units", group_cols=["store_id", "sku_id"])

    inv = inv.drop_duplicates(subset=["date", "store_id", "sku_id"], keep="last")

    return sales, inv, po, products

# ----------------------------
# Output 1: Fact Sales Store Sku Daily Table
# ----------------------------
def make_fact_sales_store_sku_daily(sales, products):

    df = sales.merge(products[["sku_id", "price", "cost"]], on="sku_id", how="left")
    df["revenue"] = df["units_sold"] * df["price"]
    df["margin_proxy"] = df["units_sold"] * (df["price"] - df["cost"])

    output = df[[
        "date","store_id","sku_id",
        "units_sold","revenue","margin_proxy",
        "promo_flag","holiday_flag","day_of_week"
    ]].copy()

    output.to_csv(OUT / "fact_sales_store_sku_daily.csv", index=False)
    return output

# ----------------------------
# Output 2: Fact Inventory Store Sku Daily
# ----------------------------
def make_fact_inventory_store_sku_daily(inv, sales):
    demand = sales[["date","store_id","sku_id","true_demand_units"]].copy()
    demand = demand.sort_values("date")
    demand["avg_daily_demand_4w"] = (
        demand.groupby(["store_id","sku_id"])["true_demand_units"]
        .transform(lambda x: x.rolling(28, min_periods=7).mean())
    )

    df = inv.merge(
        demand[["date","store_id","sku_id","avg_daily_demand_4w"]],
        on=["date","store_id","sku_id"],
        how="left"
    )

    df["stockout_flag"] = df["on_hand_units"] <= 0
    df["days_of_cover"] = df["on_hand_units"] / df["avg_daily_demand_4w"]
    df["days_of_cover"] = df["days_of_cover"].replace([np.inf, -np.inf], np.nan)

    output = df[[
        "date","store_id","sku_id",
        "on_hand_units","stockout_flag","days_of_cover"
    ]].copy()

    output.to_csv(OUT / "fact_inventory_store_sku_daily.csv", index=False)
    return output

# ----------------------------
# Output 3: Replenishment Inputs Store Sku
# ----------------------------
def make_replenishment_inputs_store_sku(sales, inv, po, products):
    max_date = sales["date"].max()
    recent = sales[sales["date"] >= (max_date - pd.Timedelta(days=56))].copy()

    demand_stats = recent.groupby(["store_id","sku_id"]).agg(
        avg_daily_demand=("true_demand_units","mean"),
        demand_std_dev=("true_demand_units","std"),
    ).reset_index()

    if "lead_time_days" not in po.columns:
        raise ValueError("purchase_orders.csv must contain 'lead_time_days'")
    lead = po.groupby(["store_id","sku_id"], as_index=False)["lead_time_days"].mean()

    latest_inv = (inv.sort_values("date").groupby(["store_id","sku_id"], as_index=False).tail(1))[
        ["store_id","sku_id","on_hand_units"]
    ]

    df = (demand_stats
          .merge(lead, on=["store_id","sku_id"], how="left")
          .merge(latest_inv, on=["store_id","sku_id"], how="left")
          .merge(products[["sku_id","category","moq_units"]], on="sku_id", how="left"))

    # fill missing basics
    df["lead_time_days"] = df["lead_time_days"].fillna(df["lead_time_days"].median())
    df["demand_std_dev"] = df["demand_std_dev"].fillna(0)
    df["on_hand_units"] = df["on_hand_units"].fillna(0)
    df["moq_units"] = df["moq_units"].fillna(1)

    # service level by category
    df["service_level_target"] = df["category"].map(SERVICE_LEVEL_BY_CATEGORY).fillna(0.95)
    df["z"] = df["service_level_target"].apply(lambda x: norm.ppf(x))

    # safety stock & reorder point
    df["safety_stock"] = df["z"] * df["demand_std_dev"] * np.sqrt(df["lead_time_days"])
    df["reorder_point"] = df["avg_daily_demand"] * df["lead_time_days"] + df["safety_stock"]

    # recommended order
    df["recommended_order_qty"] = (df["reorder_point"] - df["on_hand_units"]).clip(lower=0)

    # round up to MOQ
    df["recommended_order_qty"] = np.ceil(df["recommended_order_qty"] / df["moq_units"]) * df["moq_units"]

    df.to_csv(OUT / "replenishment_inputs_store_sku.csv", index=False)
    return df

#main method to execute the above code
def run():
    print("Loading raw data...")
    stores, sales, inv, po, products = load_all()

    print("Cleaning input tables...")
    sales, inv, po, products = clean_inputs(sales, inv, po, products)

    print("Creating Output 1: fact_sales_store_sku_daily.csv")
    fact_sales = make_fact_sales_store_sku_daily(sales, products)

    print("Creating Output 2: fact_inventory_store_sku_daily.csv")
    fact_inv = make_fact_inventory_store_sku_daily(inv, sales)

    print("Creating Output 3: replenishment_inputs_store_sku.csv")
    repl = make_replenishment_inputs_store_sku(sales, inv, po, products)

    print("\n✅ ETL complete.")
    print("Files saved in /data folder:\n")

    print(f"fact_sales_store_sku_daily.csv → {len(fact_sales):,} rows")
    print(f"fact_inventory_store_sku_daily.csv → {len(fact_inv):,} rows")
    print(f"replenishment_inputs_store_sku.csv → {len(repl):,} rows")


if __name__ == "__main__":
    run()