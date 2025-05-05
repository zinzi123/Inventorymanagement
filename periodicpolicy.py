import pandas as pd
import numpy as np
import math
import streamlit as st
from scipy.stats import norm
import matplotlib.pyplot as plt

# Streamlit UI
st.title('Probabilistic Inventory')
st.header("Periodic Review Policy",divider='orange')
# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Determine the file type and read the dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Display the uploaded dataset
    st.write("### Dataset Preview:")
    st.dataframe(df.head())
    st.markdown('<hr style="border: 1px solid orange;">', unsafe_allow_html=True)

    # SKU selection
    SKU = df['SKU_ID'].unique()
    SKU_x = st.selectbox('Select SKU', SKU)

    # User inputs for parameters with default values
    LD = st.number_input('Enter Lead Time (in days)', min_value=1, value=10)
    CSL = st.number_input('Enter Desired Service Level (e.g., 0.95 for 95%)', min_value=0.01, max_value=0.99, value=0.95)
    R = st.number_input('Enter Review Period (in days)', min_value=1, value=7)

    # Calculate safety factor (k) for the given service level using the inverse of the normal distribution CDF
    k = norm.ppf(CSL)
    st.write(f"Safety Factor (k): {k:.2f}")

    # Extract the daily demand data for the selected SKU
    df_sim = pd.DataFrame({'time': np.array(range(1, 365+1))})
    df_sim['demand'] = np.array(df.loc[df['SKU_ID'] == SKU_x].values[0][2:])

    # Calculate Average Daily Demand
    D_day = df_sim['demand'].mean()
    st.write(f"Average Daily Demand: {D_day:.2f} units")

    # Calculate Standard Deviation of Daily Demand
    sigma = df_sim['demand'].std()
    st.write(f"Standard Deviation of Daily Demand: {sigma:.2f} units")

    # Calculate Average Demand during Lead Time
    mu_ld = D_day * LD
    st.write(f"Average Demand during Lead Time: {mu_ld:.2f} units")

    # Calculate Standard Deviation during Lead Time
    sigma_ld = sigma * math.sqrt(LD)
    st.write(f"Standard Deviation during Lead Time: {sigma_ld:.2f} units")

    # Calculate Reorder Point
    s = mu_ld + k * sigma_ld
    st.write(f"Reorder Point: {s:.2f} units")

    # Calculate Order-Up-To Level
    mu_ld_R = D_day * (LD + R)
    sigma_ld_R = sigma * math.sqrt(LD + R)
    S = mu_ld_R + k * sigma_ld_R
    st.write(f"Order-Up-To Level: {S:.2f} units")

    # Initialize the inventory at the Order-Up-To Level
    initial_inventory = S
    st.write(f"Initial Inventory Level: {initial_inventory:.2f} units")
    st.markdown('<hr style="border: 1px solid orange;">', unsafe_allow_html=True)

    # Simulate Inventory Management
    inventory_levels = [initial_inventory]
    replenishments = []
    orders = [0] * 365  # Track daily orders
    total_demand = 0
    unmet_demand = 0
    demand_fulfilled = 0

    for day in range(1, 365+1):
        daily_demand = df_sim.loc[day-1, 'demand']
        total_demand += daily_demand
        current_inventory = inventory_levels[-1] - daily_demand
        
        if current_inventory < 0:
            unmet_demand -= current_inventory  # Track unmet demand
            current_inventory = 0

        demand_fulfilled += min(inventory_levels[-1], daily_demand)

        # Check if there are any scheduled replenishments for today
        if day in [replenishment[0] for replenishment in replenishments]:
            replenishment_index = [replenishment[0] for replenishment in replenishments].index(day)
            current_inventory += replenishments[replenishment_index][1]
        
        # Place order if current inventory is below reorder point
        if current_inventory < s and day % R == 0:
            replenishment_qty = S - current_inventory
            replenishments.append((day + LD, replenishment_qty))  # Schedule replenishment after lead time
            orders[day-1] = replenishment_qty

        inventory_levels.append(current_inventory)

    # Prepare data for visualization
    df_sim['inventory'] = inventory_levels[1:]
    df_sim['reorder_point'] = s
    df_sim['order_up_to_level'] = S
    df_sim['order'] = orders

    # Display the combined data table
    st.write('### Inventory Management Data')
    st.dataframe(df_sim[['time', 'demand', 'order', 'inventory']])
    st.markdown('<hr style="border: 1px solid orange;">', unsafe_allow_html=True)

    # Plot trend chart using Matplotlib
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df_sim['time'], df_sim['demand'], label='Demand', color='blue')
    ax1.plot(df_sim['time'], df_sim['inventory'], label='Inventory', color='green')
    for replenishment in replenishments:
        ax1.axvline(x=replenishment[0], color='orange', linestyle='--', linewidth=0.8)

    ax1.set_title('Inventory Management Trends')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Units')
    ax1.legend()

    st.pyplot(fig)
    st.markdown('<hr style="border: 1px solid orange;">', unsafe_allow_html=True)

    # Calculate additional metrics
    service_level = 100 - (unmet_demand / total_demand * 100)
    average_inventory = np.mean(inventory_levels)

    st.write(f"### Actual Inventory Performance")
    st.write(f"Service Level: {service_level:.2f}%")
    st.write(f"Average Inventory Level: {average_inventory:.2f} units")
    st.markdown('<hr style="border: 1px solid orange;">', unsafe_allow_html=True)

    # Prepare replenishments data for continuous timeline
    df_replenishments = pd.DataFrame({
        'Day': list(range(1, 365+1)),
        'Quantity': orders
    })

    # Display Replenishments as Bar Chart
    st.write("### Replenishments Over Time")
    st.bar_chart(df_replenishments.set_index('Day'))

else:
    st.write("Please upload a dataset to begin.")
