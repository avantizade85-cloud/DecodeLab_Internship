import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg
from plotly.subplots import make_subplots

# Load cleaned dataset
df = pd.read_csv("retail_orders_cleaned.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Create subplots dashboard
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "Sales Share by Category", 
        "Sales & Profit by Region", 
        "Sales Trend Over Time", 
        "Profit vs Sales Scatter", 
        "Sales Distribution", 
        "Sales by Customer Segment"
    ),
    specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
)

# 1. Pie Chart - Sales by Category
cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
fig.add_trace(
    gg.Pie(labels=cat_sales["Category"], values=cat_sales["Sales"], name="Category", hole=0.4),
    row=1, col=1
)

# 2. Bar Chart - Region Performance
reg_perf = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index()
fig.add_trace(
    gg.Bar(x=reg_perf["Region"], y=reg_perf["Sales"], name="Sales (Region)", marker_color="#3366cc"),
    row=1, col=2
)
fig.add_trace(
    gg.Bar(x=reg_perf["Region"], y=reg_perf["Profit"], name="Profit (Region)", marker_color="#109618"),
    row=1, col=2
)

# 3. Line Chart - Monthly Sales Trend
df_monthly = df.set_index("Order_Date").resample("ME")[["Sales", "Profit"]].sum().reset_index()
fig.add_trace(
    gg.Scatter(x=df_monthly["Order_Date"], y=df_monthly["Sales"], mode='lines+markers', name="Monthly Sales", line=dict(color="#ff9900", width=3)),
    row=1, col=3
)

# 4. Scatter Plot - Profit vs Sales
fig.add_trace(
    gg.Scatter(
        x=df["Sales"], y=df["Profit"], 
        mode='markers', 
        text=df["Category"],
        marker=dict(size=8, color=df["Profit"], colorscale="Viridis", showscale=True),
        name="Profit vs Sales"
    ),
    row=2, col=1
)

# 5. Histogram - Sales Distribution
fig.add_trace(
    gg.Histogram(x=df["Sales"], nbinsx=30, name="Sales Dist", marker_color="#990099"),
    row=2, col=2
)

# 6. Bar Chart - Customer Segment
seg_sales = df.groupby("Segment")["Sales"].sum().reset_index()
fig.add_trace(
    gg.Bar(x=seg_sales["Segment"], y=seg_sales["Sales"], name="Segment Sales", marker_color="#dc3912"),
    row=2, col=3
)

# Update layout
fig.update_layout(
    title_text="<b>Retail Orders Data Science Interactive Dashboard</b>",
    title_x=0.5,
    title_font=dict(size=20),
    template="plotly_white",
    height=850,
    showlegend=True
)

# Save to HTML
output_file = "interactive_dashboard.html"
fig.write_html(output_file)
print(f"Successfully generated interactive dashboard: {output_file}")
