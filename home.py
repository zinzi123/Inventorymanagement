import streamlit as st

# Streamlit app
st.title('G10X Inventory Management Suite')

# HTML for aesthetics
st.markdown("""
<style>
    body {
        background-color: #000000;  /* Black background */
        color: #ffffff;  /* White text color for contrast */
    }
    .section-title {
        font-size: 18px;  /* Further reduced font size */
        color: #ff8c00;  /* Subtler orange color */
        padding-bottom: 8px;
        font-weight: bold;
    }
    .section-content {
        border: 1px solid #ff8c00;  /* Subtler orange color */
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #1e1e1e; /* Dark background color for readability */
    }
    .icon {
        font-size: 40px;
        color: #ff8c00;
        vertical-align: middle;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.write("""
Welcome to the G10X Inventory Management Suite. Explore our inventory management strategies below:
""")

# Section 1: Deterministic (Constant) Demand Inventory Management
st.markdown('<div class="section-content"><div class="section-title"><span class="icon">📦</span>1. Deterministic (Constant) Demand Inventory Management</div>', unsafe_allow_html=True)
st.write("""
Ideal for stable and predictable demand environments. Key points include:
- **EOQ**: Optimal order quantity to minimize costs.
- **Replenishment Period**: Regular time interval between orders.
- **Inventory Simulation**: Simulation of inventory turnover over time with relevant visuals.
- **Cost Analysis**: Visualization of relevant cost heads monthly and yearly.
- **Cumulative COGS & Turnover**: Visuals showing the cumulative impact.

""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 2: Probabilistic Inventory Management with Continuous Review Policy
st.markdown('<div class="section-content"><div class="section-title"><span class="icon">🔄</span>2. Probabilistic Inventory Management with Continuous Review Policy</div>', unsafe_allow_html=True)

st.write("""
This model is designed for environments with fluctuating demand and requires continuous monitoring of inventory levels. Key aspects include:

- **Economic Order Quantity (EOQ)**: Determines the optimal order quantity to minimize the sum of ordering and holding costs. 
- **Reorder Point**: The inventory level at which a new order is placed to avoid stockouts, calculated considering safety stock.
- **Safety Stock**: Additional inventory held to mitigate the risk of stockouts due to demand variability or supply delays.
- **Inventory Simulation**: Simulates inventory levels over time, taking into account demand variability, reorder points, and lead times. 
- **Cost Analysis**: Provides insights into the total cost of inventory management, including ordering costs, holding costs, and the cost of stockouts.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 3: Probabilistic Inventory Management with Periodic Review Policy
st.markdown('<div class="section-content"><div class="section-title"><span class="icon">📅</span>3. Probabilistic Inventory Management with Periodic Review Policy</div>', unsafe_allow_html=True)
st.write("""
In this model, we implement a **Periodic Review Policy** to improve inventory management, especially when dealing with a large portfolio of items with varying replenishment cycles. This approach is more efficient compared to continuous review policies for diverse inventories.

Key features include:

- **Handling Multiple SKUs**: Manages inventory for various products, each with its own demand patterns and review intervals.
- **Order-Up-To Level**: Sets the target inventory level to which stock should be replenished at each review, optimizing stock levels.
- **Reorder Point**: Establishes the inventory level at which new orders are triggered to prevent stockouts, factoring in lead time and safety stock.
- **Actual Performance Measurement**: Compares simulated performance with actual metrics to assess how well the policy maintains inventory levels and avoids stockouts.

""")

st.markdown('</div>', unsafe_allow_html=True)
