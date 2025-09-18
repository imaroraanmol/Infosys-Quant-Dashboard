# Infosys-Quant-Dashboard
Cut through the market noise.🔬 An interactive dashboard to experiment with Infosys's valuation using DCF, Monte Carlo simulations, and peer analysis.
🔬 Infosys Valuation Lab: An Interactive Financial Dashboard
Ever wondered what a tech giant like Infosys is really worth? Static reports and confusing spreadsheets can only tell you so much. I built this interactive dashboard to cut through the noise and bring financial analysis to life.

This tool lets you become the analyst. Play with key financial assumptions, simulate thousands of possible outcomes, and see for yourself how the valuation of Infosys changes in real-time. Let's demystify quantitative finance together!

✨ What Can You Do Here?
This dashboard is packed with tools to let you explore Infosys's valuation from every angle:

Play with the Numbers in Real-Time 🎛️: Tweak the WACC and growth rate sliders and instantly see how these crucial inputs impact the company's intrinsic value.

See 10,000 Possible Futures 🎲: Run a Monte Carlo simulation to generate a range of potential DCF valuations. This helps you understand the optimistic, pessimistic, and most likely outcomes.

Find the Breaking Points with a Heatmap 🔥: The sensitivity analysis visualizes how the valuation holds up under different economic conditions.

Compare with the Competition 📊: See how Infosys stacks up against industry rivals like TCS, Wipro, and HCL Tech across key metrics like P/E Ratio and EV/EBITDA.

Is it a Bargain or a Bubble? ⚖️: Input the current stock price and compare it against the model's data-driven valuation to get an instant take on whether the stock might be over or undervalued.

🛠️ What's Under the Hood?
This project was built using a modern Python data science stack, all wrapped in a clean, user-friendly interface.

Web Framework: Streamlit

Data Crunching: Pandas & NumPy

Beautiful Visuals: Matplotlib, Seaborn & Plotly

Custom Theming: The dashboard features a custom color scheme to make the data pop.


Dev Environment: Ready to go with a pre-configured Docker container for easy setup.

🚀 Get It Running in 2 Minutes
Ready to start your own analysis? Just follow these simple steps.

Clone this repository:

Bash

git clone https://github.com/imaroraanmol/Infosys-Quant-Dashboard.git
cd Infosys-Quant-Dashboard
Set up your environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the magic (dependencies):

Bash

pip install -r requirements.txt
Launch the dashboard!
Your browser should open with the app ready to go.

Bash

streamlit run dashboard.py


🗺️ A Quick Tour of the Lab
Main Control Panel (INFOSYS ANALYSIS): This is your starting point. Use the sliders for WACC and Growth Rate to see the core DCF and Intrinsic Value calculations change instantly. Don't forget to pop in an exchange rate if you want to see the numbers in Rupees (₹).

The What-If Machine (Sensitivity Analysis): The heatmap shows you the landscape of possibilities. It's perfect for understanding which assumptions have the biggest impact on the final price.

The Reality Check (Peer Comparison): A company's value is relative. Use this section to see if Infosys is leading the pack or falling behind its competitors.

The Crystal Ball (Monte Carlo Simulation): This is where the real magic happens. By running thousands of simulations, you get a much richer sense of the valuation than a single number could ever provide.

The Final Verdict (Live Valuation): Put your findings to the test. Pit the model's valuation against the live market price and see what you find!

⚠️ A Friendly Heads-Up
This dashboard is a powerful tool for learning and exploration. However, it's built for educational purposes with hardcoded data and simplified models. Please remember, this is not financial advice. Always do your own extensive research before making any investment decisions.
