# Cross-Sectional Equity Alpha Model (S&P 500)

This project demonstrates a **quantitative stock ranking model** trained on historical data from **all S&P 500 stocks**.  
Rather than predicting exact prices, the model is designed to **rank stocks by expected relative performance** on each trading day.

An interactive **Streamlit dashboard** is included to explore the model’s rankings, evaluation metrics, and historical price context.

---

## 📌 What Problem Does This Model Solve?

Given all stocks in the S&P 500 on a given trading day:

**Which stocks are more likely to outperform others in the near future?**

The model answers this by:
- Assigning an **alpha score** to each stock every day
- Ranking stocks from most to least attractive
- Evaluating whether higher-ranked stocks actually performed better than lower-ranked ones

This ranking-based approach is commonly used in quantitative investing because it is more stable and practical than trying to forecast exact returns.

---

## 🧠 Model Overview

- **Universe:** All S&P 500 constituents  
- **Approach:** Cross-sectional stock ranking  
- **Model:** LightGBM (gradient boosting)  
- **Features:** Price trends, momentum, volatility, liquidity, and related market signals  
- **Neutralization:** Scores are adjusted to remove broad market and sector effects, isolating stock-specific signals  

Two scores are produced:
- **Raw Alpha Score** – direct model output  
- **Neutralized Alpha Score** – model output after removing common risk factors  

---

## 📊 How Performance Is Evaluated

The model is evaluated using **Information Coefficient (IC)**, a standard metric in quantitative finance.

- **Daily IC:** Measures how well the model ranked all stocks on a given day  
- **IC Mean:** Average ranking effectiveness over time  
- **IC IR (Mean / Std):** Consistency of the ranking signal  
- **Hit Rate:** Percentage of days where rankings worked in the correct direction  

These metrics focus on **ranking quality and consistency**, not absolute return prediction.

---

## 🖥 Interactive Dashboard (Streamlit)

The project includes an interactive Streamlit app that allows users to:

- Select a **stock ticker** and **date**
- View the stock’s **alpha score and rank** for that day
- See whether the model’s **overall ranking worked** on that date
- View a **historical price chart** for context (capped to Jan 1, 2022)
- Explore **top and bottom ranked stocks** for any trading day
- Review a **model performance snapshot** across the evaluation period

---

## 🚀 How to Run the App

### Option 1: View Online (Recommended)
If deployed on Streamlit Community Cloud, simply open the public app link list below.
URL: https://csealphamodel-qjn3xmebgiuezy8mpcwxhc.streamlit.app/

### Option 2: Run Locally
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run streamlit_recruiter_app.py
