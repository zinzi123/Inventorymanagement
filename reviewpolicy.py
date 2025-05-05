import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Function Definitions

# Unit Normal Loss Function
def unlf(k, sigma):
    result = stats.norm(0, 1).pdf(k) - k * (1 - stats.norm(0, 1).cdf(k))
    return sigma * result

# Simulation function with lead time compensated for probabilistic demand
def sim(Q, D_day, T_total, LD, sigma, reorder_point):
    df_sim = pd.DataFrame({'time': np.array(range(1, T_total + 1))})
    df_sim['demand'] = df_sim['time'].apply(lambda t: max(0, round(np.random.normal(D_day, sigma), 1)))

    # Initialize orders and inventory on hand
    orders = [0] * T_total
    ioh = [0] * T_total
    ioh[0] = round(Q, 1)
    pending_order = False

    for t in range(1, T_total):
        # Decrease IOH by daily demand
        ioh[t] = round(ioh[t - 1] - df_sim['demand'][t], 1)

        # Check for incoming orders
        if t >= LD and orders[t - LD] > 0:
            ioh[t] = round(ioh[t] + orders[t - LD], 1)
            pending_order = False  # Order has arrived

        # Place an order if IOH drops below reorder point and no pending order
        if ioh[t] < reorder_point and not pending_order:
            orders[t] = round(Q, 1)
            pending_order = True

        # Ensure no negative inventory on hand
        if ioh[t] < 0:
            ioh[t] = 0  # Set IOH to 0 to avoid negative inventory

    df_sim['order'] = orders
    df_sim['ioh'] = ioh

    return df_sim

# Function to add orange dividers
def orange_divider():
    st.markdown("<hr style='border: 2px solid orange;'>", unsafe_allow_html=True)

# Streamlit UI
st.title("Probabilistic Inventory")
st.header("Continuous Review Policy",divider='orange')

# Inputs
st.header("Input Parameters")

D = st.number_input("Total Demand (units/year)", value=2000, help="Estimated total demand for the product over a year.")
T_total = st.number_input("Total Time (days)", value=365, help="Period over which the simulation is run (typically 365 days for a yearly simulation).")
c = st.number_input("Cost of Product ($/unit)", value=50, help="Cost of producing or purchasing one unit of the product.")
c_t = st.number_input("Cost of Placing an Order ($/order)", value=500, help="Fixed cost associated with placing an order.")
h = st.number_input("Holding Cost (% unit cost per year)", value=0.25, help="Cost of holding one unit of inventory for a year, expressed as a percentage of the unit cost.")
p = st.number_input("Selling Price ($/unit)", value=75, help="Price at which the product is sold.")
LD = st.number_input("Lead Time (days)", value=5, help="Time between placing an order and receiving the inventory.")
c_s = st.number_input("Cost of Shortage ($/unit)", value=12, help="Cost incurred for each unit shortfall in inventory.")
sigma = st.number_input("Standard Deviation of Demand (units/year)", value=50, help="The variability in demand over a year.")
CSL = st.number_input("Cycle Service Level (CSL)", value=0.95,  help= "The desired service level (e.g., 95%). CSL represents the probability that inventory will be sufficient to meet customer demand during the order cycle without stockouts. A CSL of 95% means there is a 95% chance that demand will be met without running out of stock.")

if st.button("Run Simulation"):
    D_day = round(D / T_total, 1)
    sigma_day = sigma / math.sqrt(T_total)
    H = h * c

    # Calculate reorder point (s)
    mu_ld = D_day * LD
    sigma_ld = sigma_day * math.sqrt(LD)
    k = round(stats.norm(0, 1).ppf(CSL), 2)
    reorder_point = mu_ld + k * sigma_ld
    G_k = unlf(k, sigma_ld)

    # Calculate EOQ
    EOQ = round(math.sqrt((2 * D * c_t) / H), 1)

    # Plot EOQ Curves
    st.subheader("Economic Order Quantity (EOQ) Curves")
    Q_values = np.arange(1, 1001)
    order_cost = (D / Q_values) * c_t
    holding_cost = (Q_values / 2) * H
    total_cost = order_cost + holding_cost

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Q_values, order_cost, label='Order Cost', color='blue')
    ax.plot(Q_values, holding_cost, label='Holding Cost', color='red')
    ax.plot(Q_values, total_cost, label='Total Relevant Cost', color='black')
    ax.set_xlabel('Order Quantity (units/order)')
    ax.set_ylabel('Costs ($)')
    ax.set_title('Total Relevant Cost = Order Cost + Holding Cost')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 30000)  # Adjust this value based on the range you need

    plt.tight_layout()  # Adjust spacing to prevent overlap
    st.pyplot(fig)

    orange_divider()

    # Display EOQ and Reorder Point
    st.subheader("Economic Order Quantity (EOQ) and Reorder Point")
    st.write(f"The Economic Order Quantity (EOQ) is approximately {EOQ:.1f} units.")
    st.write(f"The Reorder Point is approximately {reorder_point:.1f} units.")
    st.write(f"Safety Stock is approximately {k * sigma_ld:.1f} units.")

    orange_divider()

    # Run Simulation
    df_sim1 = sim(EOQ, D_day, T_total, LD, sigma_day, reorder_point)

    # Display simulation as table
    st.subheader("Simulation Table")
    st.dataframe(df_sim1)

    # Calculate costs
    total_orders = round(df_sim1['order'].sum() / EOQ, 1)
    total_order_cost = round(total_orders * c_t, 1)
    total_holding_cost = round(df_sim1['ioh'].mean() * H * (T_total / 365), 1)

    monthly_order_cost = round(total_order_cost / 12, 1)
    monthly_holding_cost = round(total_holding_cost / 12, 1)
    total_monthly_cost = round(monthly_order_cost + monthly_holding_cost, 1)

    yearly_order_cost = total_order_cost
    yearly_holding_cost = total_holding_cost
    total_yearly_cost = yearly_order_cost + yearly_holding_cost

    # Display costs
    st.subheader("Cost Analysis")

    # Monthly Costs Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_costs = [monthly_order_cost, monthly_holding_cost, total_monthly_cost]
    cost_labels = ['Monthly Order Cost', 'Monthly Holding Cost', 'Total Monthly Cost']
    ax.bar(cost_labels, monthly_costs, color=['blue', 'red', 'black'])

    for i, v in enumerate(monthly_costs):
        ax.text(i, v + 50, f"${v:.1f}", ha='center', va='bottom')

    ax.set_title('Monthly Costs')
    ax.set_ylabel('Cost ($)')
    ax.set_ylim(0, max(monthly_costs) * 1.2)

    plt.tight_layout()
    st.pyplot(fig)

    # Yearly Costs Table
    st.subheader("Yearly Costs")
    yearly_cost_data = {
        "Cost Type": ["Yearly Order Cost", "Yearly Holding Cost", "Total Yearly Cost"],
        "Cost ($)": [round(yearly_order_cost, 1), round(yearly_holding_cost, 1), round(total_yearly_cost, 1)]
    }
    df_yearly_costs = pd.DataFrame(yearly_cost_data)
    st.table(df_yearly_costs)

    orange_divider()

    # Display Simulation Charts
    st.subheader("Simulation Charts")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    df_sim1.plot(x='time', y='demand', ax=axes[0], grid=True, color='r', title='Demand over Time')
    df_sim1.plot.scatter(x='time', y='order', ax=axes[1], grid=True, color='b', title='Orders Placed over Time')
    df_sim1.plot(x='time', y='ioh', ax=axes[2], grid=True, color='g', title='Inventory on Hand (IOH) over Time')

    plt.tight_layout()
    st.pyplot(fig)

    orange_divider()

    # Calculate cumulative values for turnover and COGS
    df_sim1['turnover'] = df_sim1['demand'].cumsum() * p
    df_sim1['COGS'] = df_sim1['demand'].cumsum() * c
    df_sim1['ordering_cost'] = (df_sim1['order'] > 0).cumsum() * c_t
    df_sim1['holding_cost'] = df_sim1['ioh'].cumsum() * H / T_total

    # Cumulative Turnover and COGS Chart
    st.subheader("Cumulative Turnover and COGS")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sim1['time'], df_sim1['turnover'], label='Cumulative Turnover', color='green')
    ax.plot(df_sim1['time'], df_sim1['COGS'], label='Cumulative COGS', color='purple')

    ax.set_title('Cumulative Turnover and Cost of Goods Sold (COGS) over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Amount ($)')
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Cumulative Cost Breakdown Chart
    st.subheader("Cumulative Cost Breakdown")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sim1['time'], df_sim1['ordering_cost'], label='Cumulative Ordering Cost', color='blue')
    ax.plot(df_sim1['time'], df_sim1['holding_cost'], label='Cumulative Holding Cost', color='red')

    ax.set_title('Cumulative Ordering and Holding Costs over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cost ($)')
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
