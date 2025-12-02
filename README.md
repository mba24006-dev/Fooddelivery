# 🍔 Food Delivery App Analysis Dashboard

A comprehensive Streamlit application for analyzing food delivery app data with machine learning predictions and interactive visualizations.

## 📊 Features

### 1. **Dashboard Overview**
- Key metrics: Total customers, orders, churn rate, and average spending
- Churn distribution visualization
- Top cities by order volume
- Customer spending and order frequency distributions

### 2. **Exploratory Data Analysis (EDA)**
- Descriptive statistics for customer-level data
- Gender and age group distributions
- Churn rate analysis by city
- Top restaurants by order count
- Delivery status distribution
- Feature correlation heatmap
- Order volume trends by day of week

### 3. **Churn Prediction Model**
- Random Forest classification model
- Model accuracy and performance metrics
- Top 10 features influencing churn
- Confusion matrix and classification report
- Model trained on customer behavior patterns

### 4. **Delivery Delay Prediction**
- Machine learning model predicting delivery delays
- Top 10 features affecting delays
- Confusion matrix visualization
- Restaurants ranked by delay rates

### 5. **Customer Segmentation**
- K-Means clustering into 3 customer segments:
  - **Loyal & Recent (VIP)**: High-value, recent customers
  - **At-Risk (Churn)**: Customers likely to churn
  - **New/Low-Value**: New or low-spending customers
- RFM (Recency, Frequency, Monetary) analysis
- Segment characteristics and churn rates

### 6. **Interactive Churn Prediction**
- Predict individual customer churn status
- Input customer details and get instant predictions
- Confidence scores for predictions

## 🚀 Deployment on Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it `food-delivery-analysis`
3. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/food-delivery-analysis.git
   cd food-delivery-analysis
   ```

### Step 2: Add Project Files

Copy all the following files to your repository:
- `streamlit_app.py` - Main Streamlit application
- `FooddeliveryAppAnalysisDataset.csv` - Dataset
- `rf_model.pkl` - Pre-trained Random Forest model
- `feature_columns.json` - Feature configuration
- `requirements.txt` - Python dependencies
- `README.md` - This file

### Step 3: Push to GitHub

```bash
git add .
git commit -m "Initial commit: Food Delivery Analysis Dashboard"
git push origin main
```

### Step 4: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository: `YOUR_USERNAME/food-delivery-analysis`
4. Select the branch: `main`
5. Set the main file path: `streamlit_app.py`
6. Click "Deploy"

Your app will be live at: `https://YOUR_USERNAME-food-delivery-analysis.streamlit.app`

## 📋 File Descriptions

| File | Size | Description |
|------|------|-------------|
| `streamlit_app.py` | 25 KB | Main Streamlit application with all visualizations and models |
| `FooddeliveryAppAnalysisDataset.csv` | 822 KB | Food delivery dataset with 6,000 orders |
| `rf_model.pkl` | 19 MB | Pre-trained Random Forest model for churn prediction |
| `feature_columns.json` | 381 B | Feature names and model metadata |
| `requirements.txt` | 100 B | Python package dependencies |

## 🛠️ Local Development

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/food-delivery-analysis.git
   cd food-delivery-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your browser to `http://localhost:8501`

## 📦 Dependencies

- **streamlit** (1.32.2) - Web app framework
- **pandas** (2.1.4) - Data manipulation
- **numpy** (1.24.3) - Numerical computing
- **matplotlib** (3.8.2) - Visualization
- **seaborn** (0.13.1) - Statistical visualization
- **scikit-learn** (1.3.2) - Machine learning

## 📊 Dataset Overview

**FooddeliveryAppAnalysisDataset.csv**
- **Records**: 6,000 orders
- **Customers**: Aggregated customer-level metrics
- **Date Range**: August 2023 - July 2025
- **Features**: 21 columns including:
  - Customer demographics (age, gender, city)
  - Order details (restaurant, category, price, quantity)
  - Delivery information (status, timing)
  - Customer engagement (loyalty points, churn status)

## 🎯 Key Insights

### Model Performance
- **Churn Prediction Accuracy**: ~49.4%
- **Top Churn Predictors**: Days since last order, order frequency, total spent
- **Delivery Delay Prediction**: Trained on order features

### Customer Segments
1. **VIP Customers** (~33%): Recent, high-value customers
2. **At-Risk Customers** (~33%): Inactive, likely to churn
3. **New/Low-Value** (~34%): New or low-spending customers

### Churn Statistics
- **Overall Churn Rate**: ~50% of customers are inactive
- **Highest Churn Cities**: Varies by region
- **Key Retention Factors**: Order frequency, loyalty points, recency

## 🔧 Customization

### Modify the Dataset
Replace `FooddeliveryAppAnalysisDataset.csv` with your own data. Ensure it has the same column structure.

### Retrain the Model
To retrain the churn prediction model:

```python
python3 << 'EOF'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load and prepare data
df = pd.read_csv('FooddeliveryAppAnalysisDataset.csv')
# ... data preprocessing ...

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
EOF
```

### Modify Colors and Styling
Edit the color variables in `streamlit_app.py`:
- `#2ecc71` - Green (Active/Success)
- `#e74c3c` - Red (Inactive/Warning)
- `#3498db` - Blue (Primary)
- `#f39c12` - Orange (Secondary)

## 📝 Usage Examples

### Dashboard Overview
View high-level metrics and key performance indicators at a glance.

### Exploratory Data Analysis
Explore data distributions, correlations, and trends across different dimensions.

### Churn Prediction
Understand which factors influence customer churn and review model performance metrics.

### Interactive Prediction
Input specific customer details to get instant churn predictions with confidence scores.

## 🐛 Troubleshooting

### Model File Too Large
If `rf_model.pkl` (19 MB) exceeds GitHub's file size limits:
1. Use Git LFS (Large File Storage)
2. Or retrain a smaller model with fewer estimators

### Data Not Loading
Ensure `FooddeliveryAppAnalysisDataset.csv` is in the same directory as `streamlit_app.py`.

### Import Errors
Run `pip install -r requirements.txt` to ensure all dependencies are installed.

## 📧 Support

For issues or questions:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the [scikit-learn documentation](https://scikit-learn.org)
3. Open an issue in your GitHub repository

## 📄 License

This project is provided as-is for educational and analytical purposes.

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Pandas Documentation](https://pandas.pydata.org)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Matplotlib Documentation](https://matplotlib.org)

---

**Created**: December 2, 2025  
**Version**: 2.0  
**Status**: Production Ready ✅