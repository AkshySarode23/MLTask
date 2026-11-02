import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/content/task1_marketplace_delivery_sla_raw(in).csv")

print("Original Dataset Shape:", df.shape)
print("\nOriginal Dataset:")
print(df.head())

# 2. CLEAN COLUMN NAMES

df.columns = df.columns.str.strip().str.lower()

# 3. HANDLE MIXED DATE FORMATS

def parse_mixed_dates(date_series):

    try:
        # Works with Pandas 2.0 and newer
        return pd.to_datetime(
            date_series,
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

    except (TypeError, ValueError):
        # Fallback for older Pandas versions
        return date_series.apply(
            lambda value: pd.to_datetime(
                value,
                dayfirst=True,
                errors="coerce"
            )
        )


date_columns = [
    "order_date",
    "shipped_at",
    "delivered_at",
    "updated_at"
]

for column in date_columns:
    df[column] = parse_mixed_dates(df[column])


print("\nDate Conversion Check:")

for column in date_columns:
    print(
        column,
        "Min:", df[column].min(),
        "Max:", df[column].max()
    )

# 4. REMOVE DUPLICATE ORDER RECORDS
# Keep the latest update for each order

df = df.sort_values(
    by="updated_at",
    na_position="first"
)

df = df.drop_duplicates(
    subset="order_id",
    keep="last"
)

print("\nAfter Removing Duplicate Orders:", df.shape)

# 5. REMOVE TEST ORDERS

df["is_test_order"] = (
    df["is_test_order"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[
    df["is_test_order"].isin(["no", "false", "0", "n"])
].copy()

print("After Removing Test Orders:", df.shape)


# 6. STANDARDIZE COURIER NAMES

df["courier"] = (
    df["courier"]
    .astype(str)
    .str.strip()
)

courier_key = (
    df["courier"]
    .str.lower()
    .str.replace(r"\s+", "", regex=True)
)

courier_mapping = {
    "shipquick": "ShipQuick",
    "fastbee": "FastBee",
    "bluedart": "Blue Dart"
}

df["courier"] = courier_key.map(courier_mapping).fillna(df["courier"])

# 7. CONVERT PROMISED DAYS TO NUMERIC

df["promised_days"] = pd.to_numeric(
    df["promised_days"],
    errors="coerce"
)


# 8. FILTER MAY 2026 DELIVERIES

start_date = pd.Timestamp("2026-05-01")
end_date = pd.Timestamp("2026-06-01")

may_df = df[
    (df["delivered_at"] >= start_date) &
    (df["delivered_at"] < end_date)
].copy()

print("\nMay 2026 Delivered Orders:", may_df.shape[0])

print("\nMay Delivery Records:")
print(
    may_df[
        ["order_id", "shipped_at", "delivered_at",
         "courier", "promised_days"]
    ]
)


# 9. CALCULATE ACTUAL TRANSIT DAYS

may_df["actual_transit_days"] = (
    may_df["delivered_at"] - may_df["shipped_at"
]).dt.total_seconds() / (24 * 60 * 60)


# 10. REMOVE INVALID OR INELIGIBLE RECORDS

eligible_df = may_df.dropna(
    subset=[
        "order_id",
        "courier",
        "shipped_at",
        "delivered_at",
        "promised_days",
        "actual_transit_days"
    ]
).copy()

# Exclude negative promised days and negative transit durations
eligible_df = eligible_df[
    (eligible_df["promised_days"] >= 0) &
    (eligible_df["actual_transit_days"] >= 0)
].copy()

print("\nEligible Orders:", eligible_df.shape[0])


# 11. CALCULATE ON-TIME DELIVERY

eligible_df["on_time_flag"] = (
    eligible_df["actual_transit_days"] <=
    eligible_df["promised_days"]
).astype(int)

eligible_df["delivery_status"] = (
    eligible_df["on_time_flag"]
    .map({
        1: "On Time",
        0: "Late"
    })
)

print("\nOrder-Level Delivery Results:")
print(
    eligible_df[
        ["order_id", "courier", "promised_days",
         "actual_transit_days", "delivery_status"]
    ]
)

# 12. COURIER-LEVEL PERFORMANCE SUMMARY

summary = (
    eligible_df
    .groupby("courier")
    .agg(
        eligible_orders=("order_id", "nunique"),
        on_time_orders=("on_time_flag", "sum")
    )
    .reset_index()
)

summary["on_time_rate"] = (
    summary["on_time_orders"] /
    summary["eligible_orders"]
)

summary["on_time_rate_percentage"] = (
    summary["on_time_rate"] * 100
).round(2)

summary = summary.sort_values(
    by="on_time_rate",
    ascending=False
)

print("\n======================================")
print("COURIER DELIVERY PERFORMANCE")
print("======================================")

print(
    summary[
        [
            "courier",
            "eligible_orders",
            "on_time_orders",
            "on_time_rate_percentage"
        ]
    ].to_string(index=False)
)

# 13. CHART 1: ON-TIME DELIVERY RATE BY COURIER

if not summary.empty:

    plt.figure(figsize=(9, 5))

    plt.bar(
        summary["courier"],
        summary["on_time_rate_percentage"]
    )

    plt.title("Courier-wise On-Time Delivery Rate - May 2026")
    plt.xlabel("Courier")
    plt.ylabel("On-Time Delivery Rate (%)")

    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for index, value in enumerate(
        summary["on_time_rate_percentage"]
    ):
        plt.text(
            index,
            value + 2,
            str(value) + "%",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig("courier_on_time_rate.png")
    plt.show()

else:

    print("\nNo eligible orders available for Chart 1.")

# 14. CHART 2: ELIGIBLE ORDERS VS ON-TIME ORDERS

if not summary.empty:

    x = range(len(summary))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        [position - width / 2 for position in x],
        summary["eligible_orders"],
        width=width,
        label="Eligible Orders"
    )

    plt.bar(
        [position + width / 2 for position in x],
        summary["on_time_orders"],
        width=width,
        label="On-Time Orders"
    )

    plt.xticks(
        list(x),
        summary["courier"]
    )

    plt.title("Eligible Orders vs On-Time Orders - May 2026")
    plt.xlabel("Courier")
    plt.ylabel("Number of Orders")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("courier_order_comparison.png")
    plt.show()

else:

    print("\nNo eligible orders available for Chart 2.")


# ------------------------------------------------------------
# 15. EXPORT RESULTS
# ------------------------------------------------------------

summary.to_csv(
    "courier_delivery_performance_may_2026.csv",
    index=False
)

eligible_df.to_csv(
    "may_2026_eligible_delivery_orders.csv",
    index=False
)

print("\n======================================")
print("ANALYSIS COMPLETED")
print("======================================")
