import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from IPython.display import display
import matplotlib.pyplot as plt



st.title("INFOSYS ANALYSIS")
#st.markdown("### DCF and Monte Carlo Simulation Analysis on Infosys")
#exchange = st.number_input("Enter Exchange Rate (Default: 85.26)", min_value=0.0, value=85.26, format="%.2f")
exchange_rate = st.number_input("USD price in (₹)", min_value=0.0, format="%.2f")
data={
        "Company":['Infosys','Wipro','TCS','HCL Tech'],
        "Market Price": [1922,305,4170,1911],
        'EPS':[63.29,20.82,125.88,57.86],
        'Market Cap':[7981280000000,3194450000000,15088510000000,5186760000000],
        'Debt':[83590000000,1646490000000,80210000000,57560000000],
        'Cash':[147860000000,969530000000,90160000000,94560000000],
        'EBITDA':[364250000000,1677580000000,642930000000,241980000000],
        'Revenue':[1536700000000, 8976030000000,2408930000000,1099130000000 ]
}



    
def calculate_terminal_value(last_fcf,terminal_growth,discount_rate):
    return last_fcf*(1+terminal_growth)/(discount_rate-terminal_growth)
    
def Calculate_dcf(fcf,wacc):
    dcf=[fcf/(1+wacc)**i for i,fcf in enumerate(fcf,1)]
    #st.write(dcf)
    return sum(dcf)


#slider
st.markdown("### **WACC %(Weighted average cost of capital)**")
wacc=st.slider(" WACC %",10.00,14.50,11.54)


st.subheader("DCF")
forecasted_fcfs=[3095448259.1053576, 3324705039.8328905, 3570941161.5508423, 3835414157.4914017, 4119474697.0002775]
dcf=Calculate_dcf(fcf=forecasted_fcfs,wacc=wacc/100)
#st.write(f'The (DCF) of Infosys over 5 year is: ${dcf:,.2f}')
color="green"
convert_to_inr = st.checkbox("Convert to INR")
if convert_to_inr:
    st.markdown(f"### **The (DCF) of Infosys over 5 year is: <span style='color:{color}'>{dcf*exchange_rate}₹</span>**", unsafe_allow_html=True)
else:
    st.markdown(f"### **The (DCF) of Infosys over 5 year is: <span style='color:{color}'>{dcf}$</span>**", unsafe_allow_html=True)

st.markdown("### **Growth Rate % over a long period of time**")
growth_rate=st.slider('Growth Rate %',1.00,6.50,3.00)

st.subheader('Intrinsic value')
terminal_value=terminal_value=calculate_terminal_value(forecasted_fcfs[-1],growth_rate/100,wacc/100)
sensitivity_results=dcf+terminal_value
#st.write(f'The intrinsic Value of infosys is:${sensitivity_results:,.2f}')
if convert_to_inr:
    st.markdown(f"### **The intrinsic Value of infosys is: <span style='color:{color}'>{sensitivity_results*exchange_rate}₹</span>**", unsafe_allow_html=True)
else:
    st.markdown(f"### **The intrinsic Value of infosys is: <span style='color:{color}'>{sensitivity_results}$</span>**", unsafe_allow_html=True)
fcf = forecasted_fcfs
################################################################################
#Sensitivity Analysis


wacc_values = np.linspace(0.10, 0.145, 7)  # 10% to 14.5% with 7 steps
growth_rate_values = np.linspace(0.01, 0.065, 7)  # 1% to 6.5% with 7 steps

     # Ensure forecasted FCFs are defined

# Create Sensitivity DataFrame
sensitivity_df = pd.DataFrame(index=growth_rate_values, columns=wacc_values)

for g in growth_rate_values:
    for w in wacc_values:
              # Adjust WACC - Growth Rate
            ###########
        dcf=Calculate_dcf(fcf,w)
        terminal_value=terminal_value=calculate_terminal_value(fcf[-1],g,w)
        intrinsic_value = dcf+terminal_value
            ############
        sensitivity_df.loc[g, w] = intrinsic_value

sensitivity_df = sensitivity_df.astype(float)

# Convert index and column labels to percentage format
sensitivity_df.index = [f"{g*100:.1f}%" for g in growth_rate_values]  # Growth Rate in %
sensitivity_df.columns = [f"{w*100:.1f}%" for w in wacc_values]  # WACC in %

# Plot Sensitivity Analysis Heatmap
st.markdown("## **Sensitivity Analysis: Impact of WACC & Growth Rate on Intrinsic Value in $**")

plt.figure(figsize=(14, 8))  # Increase figure size for spacing
ax = sns.heatmap(
        sensitivity_df, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f",  # Show values with 2 decimal places
        linewidths=1,  # More space between cells
        annot_kws={"fontsize": 10}  # Reduce annotation size
    )

# Adjust labels and title
plt.xlabel("WACC (%)", fontsize=14)
plt.ylabel("Growth Rate (%)", fontsize=14)
plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels for better visibility
plt.yticks(rotation=0, fontsize=12)
plt.title("Sensitivity Analysis: Intrinsic Value Heatmap", fontsize=16)

# Display in Streamlit
st.pyplot(plt)
    # Extract insights from the sensitivity analysis
min_value = sensitivity_df.min().min()  # Find the lowest intrinsic value
max_value = sensitivity_df.max().max()  # Find the highest intrinsic value

# Find the corresponding WACC and Growth Rate for min/max values
min_coords = sensitivity_df.stack().idxmin()
max_coords = sensitivity_df.stack().idxmax()

# Calculate sensitivity trend
wacc_sensitivity = sensitivity_df.diff(axis=1).mean().mean()  # Avg change across WACC
growth_sensitivity = sensitivity_df.diff(axis=0).mean().mean()  # Avg change across Growth Rate

# Generate insights
st.markdown("### 🔍 **Key Insights from Sensitivity Analysis**")

st.write(f"💡 **Best Case:** Highest intrinsic value of **{max_value:,.2f}$** or **{max_value*exchange_rate:,.2f}₹** occurs at WACC = **{max_coords[1]}** and Growth Rate = **{max_coords[0]}**.")
st.write(f"⚠️ **Worst Case:** Lowest intrinsic value of **{min_value:,.2f}$** or **{min_value*exchange_rate:,.2f}₹** occurs at WACC = **{min_coords[1]}** and Growth Rate = **{min_coords[0]}**.") 

if wacc_sensitivity < 0:
    st.write("📉 **Observation:** Intrinsic value drops significantly as WACC increases, indicating **high sensitivity** to discount rates.")
else:
    st.write("📈 **Observation:** Intrinsic value remains relatively stable with changes in WACC, indicating **low sensitivity** to discount rates.")

if growth_sensitivity > 0:
    st.write("📊 **Growth Effect:** Higher growth rates increase intrinsic value, reinforcing the importance of strong revenue growth.")
else:
    st.write("📊 **Growth Effect:** Growth rate changes have limited impact on intrinsic value, suggesting other factors are driving valuation.")

st.write("🧐 **Takeaway:** If your estimated growth rate is realistic, aim for a **lower WACC** to maximize intrinsic value. A higher WACC significantly reduces valuation, so financial risk and capital costs must be managed carefully.")

#########################################################################

##############################################################
# User-selected WACC and Growth Rate (if they want a custom case)
m_wacc = st.slider("Select WACC", min_value=10.00, max_value=15.00, value=11.75, step=0.01)
m_growth = st.slider("Select Growth Rate", min_value=1.00, max_value=6.50, value=4.50, step=0.01)
custom_wacc=m_wacc/100
custom_growth=m_growth/100
# Calculate intrinsic value for the custom case
custom_intrinsic_value = Calculate_dcf(fcf, custom_wacc)+calculate_terminal_value(fcf[-1],custom_growth,custom_wacc)

# Display Results
st.markdown("### 🔢 **Scenario Analysis & Custom Case**")

# Scenario dictionary
scenarios = {
    "Bullish": {"WACC": 0.10, "Growth Rate": 0.05, "Intrinsic Value in $": Calculate_dcf(fcf, 0.10 )+calculate_terminal_value(fcf[-1],0.05,0.10)},
    "Neutral": {"WACC": 0.125, "Growth Rate": 0.04, "Intrinsic Value in $": Calculate_dcf(fcf, 0.125 )+calculate_terminal_value(fcf[-1],0.04,0.125)},
    "Bearish": {"WACC": 0.14, "Growth Rate": 0.02, "Intrinsic Value in $": Calculate_dcf(fcf, 0.14 )+calculate_terminal_value(fcf[-1],0.02, 0.14)},
    "Custom": {"WACC": custom_wacc, "Growth Rate": custom_growth, "Intrinsic Value in $": custom_intrinsic_value}
}

# Convert to DataFrame for display
df_scenarios = pd.DataFrame.from_dict(scenarios, orient="index")
st.dataframe(df_scenarios.style.format({"Intrinsic Value": "{:,.2f}"}))

# Key Insights
st.write("### 🔍 **Key Insights**")
st.write(f"📊 If your selected WACC is **{custom_wacc:.2%}** and Growth Rate is **{custom_growth:.2%}**, the estimated intrinsic value is **{custom_intrinsic_value:,.2f}$** or **{custom_intrinsic_value*exchange_rate:,.2f}₹**.")
if custom_wacc>0.10:
    st.write(f"⚖️ As WACC increases from **0.10 to {custom_wacc:.2f}**, the intrinsic value decreases. This reflects higher discount rates reducing present value of cash flows.")

    
# Additional Interpretation
if custom_wacc < 0.115:
    st.success("✅ Favorable financing conditions! A lower WACC improves valuation.")
elif 0.115 <= custom_wacc <= 0.125:
    st.warning("⚖️ Moderately balanced risk-reward tradeoff.")
else:
    st.error("🚨 Higher WACC leads to lower intrinsic value. Consider cost management strategies.")

##############################################################

#peer analysis


st.title("Peer Comparison")
data['Market Price']= pd.to_numeric(data['Market Price'], errors='coerce')
data['EPS']=pd.to_numeric(data['EPS'], errors='coerce')
data['P/E Ratio'] = data['Market Price'] / data['EPS']
data['Market Cap']= pd.to_numeric(data['Market Cap'], errors='coerce')
data['Debt']= pd.to_numeric(data['Debt'], errors='coerce')
data['Cash']= pd.to_numeric(data['Cash'], errors='coerce')
data['Enterprise Value'] = data['Market Cap'] + data['Debt'] - data['Cash']
data['EV/EBITDA'] = data['Enterprise Value'] / data['EBITDA']
#display(data[['Market Price', 'P/E Ratio' ]])
data = pd.DataFrame(data)

# Sidebar Widgets
#st.header("Filters and Options")
#show_all = st.checkbox("Show All Peer Review", value=False)  # Default: False
selected_companies = st.multiselect(
    "Select Companies:", options=data['Company'], default=data['Company']
)

# Filter Data
filtered_data = data[data['Company'].isin(selected_companies)]

# Function to dynamically generate analysis for each metric
# Function to convert large numbers to a more readable format (M, B, T)
def format_value(value):
    if value >= 1_000_000_000_000:
        return f"₹{value / 1_000_000_000_000:.2f} T"  # Trillions
    elif value >= 1_000_000_000:
        return f"₹{value / 1_000_000_000:.2f} B"  # Billions
    elif value >= 1_000_000:
        return f"₹{value / 1_000_000:.2f} M"  # Millions
    else:
        return f"₹{value:.2f}"  # Less than a million

# Modify the dynamic analysis function to use this formatting
def dynamic_analysis(metric, filtered_data):
    max_val = filtered_data[metric].max()
    min_val = filtered_data[metric].min()
    mean_val = filtered_data[metric].mean()

    analysis = f"### {metric} Analysis:"

    # Convert values to a more readable format
    formatted_max_val = format_value(max_val)
    formatted_min_val = format_value(min_val)
    formatted_mean_val = format_value(mean_val)

    # Check if the metric is Market Price, Enterprise Value, or Market Cap (higher values are better)
    if metric in ["Market Price", "Enterprise Value", "Market Cap"]:
        highest = filtered_data[filtered_data[metric] == max_val]
        lowest = filtered_data[filtered_data[metric] == min_val]

        # Handle case where there might be more than one company with the same value
        highest_companies = ", ".join(highest['Company'].values)
        lowest_companies = ", ".join(lowest['Company'].values)

        analysis += f"\n- 📈 {highest_companies} has the highest {metric} at {formatted_max_val}, indicating a strong market presence or premium value."
        analysis += f"\n- 📉{lowest_companies} has the lowest {metric} at {formatted_min_val}, suggesting it might be undervalued or have a smaller market presence."
        
    # Check if the metric is P/E Ratio, EV/EBITDA, or EPS (lower or higher values have specific context)
    elif metric in ["P/E Ratio", "EV/EBITDA", "EPS"]:
        highest = filtered_data[filtered_data[metric] == max_val]
        lowest = filtered_data[filtered_data[metric] == min_val]

        # Handle case where there might be more than one company with the same value
        highest_companies = ", ".join(highest['Company'].values)
        lowest_companies = ", ".join(lowest['Company'].values)

        analysis += f"\n- 📈{highest_companies} has the highest {metric} at {formatted_max_val}, indicating a premium valuation or high investor expectations."
        analysis += f"\n- 📉{lowest_companies} has the lowest {metric} at {formatted_min_val}, suggesting potential undervaluation or lower growth expectations."

    else:
        # For other metrics, find the company with the highest and lowest values
        highest = filtered_data[filtered_data[metric] == max_val]
        lowest = filtered_data[filtered_data[metric] == min_val]

        # Handle case where there might be more than one company with the same value
        highest_companies = ", ".join(highest['Company'].values)
        lowest_companies = ", ".join(lowest['Company'].values)

        analysis += f"\n- {highest_companies} has the highest {metric} with a value of {formatted_max_val}."
        analysis += f"\n- {lowest_companies} has the lowest {metric} with a value of {formatted_min_val}."

    # Add average value in readable format
    analysis += f"\n- 📶The average value for {metric} is {formatted_mean_val}, providing a general sense of industry trends based on evaluation done among peers."

    return analysis


# Function to display selected metrics and their dynamic analysis
def display_selected_metrics_analysis(selected_metrics, filtered_data):
    st.subheader("Peer Comparison for Selected Metrics")
    for metric in selected_metrics:
        st.write(f"### {metric}")
        
        # Plot the bar chart
        fig = px.bar(
            filtered_data,
            x='Company',
            y=metric,
            color='Company',
            title=f"Comparison for {metric}",
            labels={metric: metric}
        )
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        st.plotly_chart(fig)

        # Dynamic Analysis for the selected metric
        analysis = dynamic_analysis(metric, filtered_data)
        st.write(analysis)

        # Statistical Summary
        st.sidebar.write(f"Average {metric}: ₹{filtered_data[metric].mean():.2f}")
        st.sidebar.write(f"Max {metric}: ₹{filtered_data[metric].max():.2f}")
        st.sidebar.write(f"Min {metric}: ₹{filtered_data[metric].min():.2f}")

# Sidebar: Select Metrics
selected_metrics = st.multiselect(
    "Select Metrics for Peer Comparison:",
    options=data.columns[1:],  # Exclude 'Company' column
    default=["Market Price", "P/E Ratio", "EV/EBITDA", "Enterprise Value"]
)

# Display analysis based on selected metrics
if selected_metrics:
    display_selected_metrics_analysis(selected_metrics, filtered_data)
else:
    st.write("Please select at least one metric for comparison.")

# Raw Data Toggle
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data in ₹")
    st.write(filtered_data)


st.title("Monte Carlo Simulation for DCF with Interactive Inputs")
wacc1=st.slider(" WACC %",10.00,15.0,11.54)
wacc=wacc1
growth1= st.slider("Select Growth Rate", min_value=1.00, max_value=6.50, value=4.00, step=0.001)
growth=growth1
st.sidebar.header("Inputs for Simulation")

WACC_mean = st.sidebar.number_input("Mean WACC (%)", min_value=0.0, max_value=100.0, value=11.53, step=0.0001,format="%.4f") / 100
WACC_std = st.sidebar.number_input("WACC Std Dev (%)", min_value=0.0, max_value=100.0, value=1.154, step=0.0001,format="%.4f") / 100
Terminal_Growth_mean = st.sidebar.number_input("Mean Terminal Growth (%)", min_value=0.0, max_value=100.0, value=3.5, step=0.0001,format="%.4f") / 100
Terminal_Growth_std = st.sidebar.number_input("Terminal Growth Std Dev (%)", min_value=0.0, max_value=100.0, value=1.443, step=0.0001,format="%.4f") / 100
num_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=1000, step=1000)

# Input for forecasted free cash flows
#forecasted_fcfs = st.sidebar.text_area(
    #"Forecasted Free Cash Flows")
#forecasted_fcfs = [float(x.strip()) for x in forecasted_fcfs.split(",")]

#result
st.header("Monte Carlo Simulation Results")

st.info("ℹ️ Open the **sidebar** for more customizing Monte Carlo simulation parameters!", icon="ℹ️")

simulated_values = []
for _ in range(num_simulations):
    # Sample WACC and terminal growth rate
    wacc_sample = np.random.normal(WACC_mean, WACC_std)
    growth_sample = np.random.normal(Terminal_Growth_mean, Terminal_Growth_std)

    if wacc_sample <= growth_sample:
        continue

    
    terminal_value = calculate_terminal_value(forecasted_fcfs[-1], growth_sample, wacc_sample)
    dcf_value = Calculate_dcf(forecasted_fcfs, wacc_sample) + (terminal_value / (1 + wacc_sample)**len(forecasted_fcfs))
    simulated_values.append(dcf_value)

simulated_values = np.array(simulated_values)


mean_value = np.mean(simulated_values)
percentile_5 = np.percentile(simulated_values, 5)
percentile_95 = np.percentile(simulated_values, 95)
#######################################################################
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="📌 Mean DCF Value", value=f"${mean_value:,.2f}", delta="Expected Valuation")

with col2:
    st.metric(label="📉 5th Percentile", value=f"${percentile_5:,.2f}", delta="Lower Bound Risk")

with col3:
    st.metric(label="📈 95th Percentile", value=f"${percentile_95:,.2f}", delta="Upper Bound Potential")

st.markdown("---")


#######################################################################################
st.write(f"**✮Mean DCF Value:** ${mean_value:,.2f}.Mean value represents the expected or average DCF value of the overall company's including terminal value discounted, based on the simulation.It is the central tendency of all the simulated outcomes")
st.write(f"**🚨5th Percentile DCF Value:** ${percentile_5:,.2f}. This suggests that there is a small probability that the company’s valuation could fall significantly lower than the mean indicating a 5% chance. This could indicate pessimistic market conditions, poor financial performance, or other risk factors.")
st.write(f"**🚀95th Percentile DCF Value:** ${percentile_95:,.2f}.The 95th percentile represents the upper bound of the simulation, indicating a 5% chance that the DCF value will exceed this amount.The most optimistic scenarios, the company's valuation could be much higher, reflecting factors like strong revenue growth, market dominance, or favorable external conditions.")

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(simulated_values, bins=50, kde=True, color='blue', alpha=0.6)
ax.axvline(mean_value, color='red', linestyle='--', label=f"Mean Value: ${mean_value:,.2f}")
ax.axvline(percentile_5, color='green', linestyle='--', label=f"5th Percentile: ${percentile_5:,.2f}")
ax.axvline(percentile_95, color='green', linestyle='--', label=f"95th Percentile: ${percentile_95:,.2f}")
ax.set_title("Monte Carlo Simulation of DCF Values", fontsize=14, fontweight='bold')
ax.set_xlabel("DCF Value (US Dollar)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend()

st.pyplot(fig)

scenario = st.selectbox("Select Scenario:", ["Bullish 🚀", "Neutral🔸", "Bearish〽️"])

if scenario == "Bullish 🚀":
    
    wacc_sample = np.random.normal(WACC_mean - 0.01, WACC_std)
    growth_sample = np.random.normal(Terminal_Growth_mean + 0.005, Terminal_Growth_std)

elif scenario == "Bearish〽️":
    
    wacc_sample = np.random.normal(WACC_mean + 0.01, WACC_std)
    growth_sample = np.random.normal(Terminal_Growth_mean - 0.005, Terminal_Growth_std)

else:  # Neutral
    wacc_sample = np.random.normal(WACC_mean, WACC_std)
    growth_sample = np.random.normal(Terminal_Growth_mean, Terminal_Growth_std)

mean_value1=mean_value
percentile_950=percentile_95

convert_to_inr1 = st.checkbox("Convert to ₹")
#exchange_rate = st.number_input("USD price in (₹)", min_value=0.0, format="%.2f")
if convert_to_inr1:
     # Example exchange rate, can be dynamic
    mean_value *= exchange_rate
    percentile_5 *= exchange_rate
    percentile_95 *= exchange_rate
    currency_symbol = "₹"
else:
    currency_symbol = "$"

st.write(f"**Mean DCF Value:** {currency_symbol}{mean_value:,.2f}")
st.write(f"**5th Percentile DCF Value:** {currency_symbol}{percentile_5:,.2f}")
st.write(f"**95th Percentile DCF Value:** {currency_symbol}{percentile_95:,.2f}")


# Define Monte Carlo estimated value (Replace with your actual calculated value)
outstanding_shares=4139950635
monte_carlo_value = (mean_value1*exchange_rate)/outstanding_shares  # Use your Monte Carlo simulation's mean DCF value

st.title("Infosys Stock Valuation")
st.subheader("Enter the Current Market Price of Infosys")

# User input for current market price
live_price = st.number_input("Current Market Price (₹)", min_value=0.0, format="%.2f")
if exchange_rate>0:
    if live_price > 0:
    # Determine valuation status based on a 10% threshold
        threshold = 0.1  # 10% deviation
        if live_price < (1 - threshold) * monte_carlo_value:
            valuation_status = "Undervalued when compared with mean DCF value"
            color = "green"
        elif live_price > (1 + threshold) * monte_carlo_value:
            valuation_status = "Overvalued when compared with mean DCF value"
            color = "red"
        else:
            valuation_status = "Fairly Valued when compared with mean DCF value"
            color = "blue"

    # Display results
        st.subheader("Live Market Price vs Monte Carlo Valuation")
        st.write(f"📌 **Live Market Price:** ₹{live_price:,.2f}")
        st.write(f"📌 **Monte Carlo Estimated Value:** ₹{monte_carlo_value:,.2f}")
        st.markdown(f"### **Status: <span style='color:{color}'>{valuation_status}</span>**", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid market price above ₹0.")


    monte_carlo_value = (percentile_950*exchange_rate)/outstanding_shares
    if live_price > 0:
    # Determine valuation status based on a 10% threshold
        threshold = 0.1  # 10% deviation
        if live_price < (1 - threshold) * monte_carlo_value:
            valuation_status = "Undervalued when compared with Best case scenario"
            color = "green"
        elif live_price > (1 + threshold) * monte_carlo_value:
            valuation_status = "Overvalued when compared with Best case scenario"
            color = "red"
        else:
            valuation_status = "Fairly Valued when compared with Best case scenario"
            color = "blue"

    # Display results
        st.subheader("Live Market Price vs Best Case Monte Carlo Valuation")
        st.write(f"📌 **Live Market Price:** ₹{live_price:,.2f}")
        st.write(f"📌 **Monte Carlo Estimated Value:** ₹{monte_carlo_value:,.2f}")
        st.markdown(f"### **Status: <span style='color:{color}'>{valuation_status}</span>**", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid market price above ₹0.")
else:
    st.warning("Enter Valid USD to ₹ conversion rate in above input tab.")
