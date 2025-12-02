import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Food Delivery Analysis Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Current date for recency calculations
CURRENT_DATE = datetime(2025, 12, 2)

@st.cache_data
def load_and_clean_data():
    """Load and clean the dataset."""
    try:
        df = pd.read_csv('FooddeliveryAppAnalysisDataset.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure FooddeliveryAppAnalysisDataset.csv is in the project directory.")
        return None, None
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle date columns
    date_cols = ['signup_date', 'order_date', 'last_order_date', 'rating_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].apply(lambda x: min(x, CURRENT_DATE) if pd.notna(x) else x)
    
    # Handle logical inconsistencies
    inconsistent_mask = df['order_date'] < df['signup_date']
    df.loc[inconsistent_mask, 'signup_date'] = df.loc[inconsistent_mask, 'order_date']
    
    # Handle missing values
    df['rating'] = df['rating'].fillna(df['rating'].median())
    df['payment_method'] = df['payment_method'].fillna('Unknown')
    df['age'] = df['age'].fillna('Unknown')
    
    # Create customer-level features
    customer_df = df.groupby('customer_id').agg(
        last_order_date=('last_order_date', 'max'),
        signup_date=('signup_date', 'min'),
        order_frequency=('order_frequency', 'max'),
        total_spent=('price', 'sum'),
        loyalty_points=('loyalty_points', 'max'),
        churned=('churned', 'first'),
        age=('age', 'first'),
        gender=('gender', 'first'),
        city=('city', 'first'),
    ).reset_index()
    
    # Calculate recency and tenure
    customer_df['days_since_last_order'] = (CURRENT_DATE - customer_df['last_order_date']).dt.days
    customer_df['customer_tenure_days'] = (customer_df['last_order_date'] - customer_df['signup_date']).dt.days
    
    # Create order-level features
    df['delivery_delay'] = df['delivery_status'].apply(lambda x: 1 if x in ['Delayed', 'Cancelled'] else 0)
    df['order_hour'] = df['order_date'].dt.hour
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    
    return df, customer_df

@st.cache_resource
def load_model():
    """Load the pre-trained Random Forest model."""
    try:
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_columns.json', 'r') as f:
            feature_info = json.load(f)
        return model, feature_info
    except FileNotFoundError:
        st.error("Model files not found. Please ensure rf_model.pkl and feature_columns.json are in the project directory.")
        return None, None

# Load data and model
order_df, customer_df = load_and_clean_data()
model, feature_info = load_model()

if order_df is None or customer_df is None or model is None:
    st.stop()

# Sidebar navigation
st.sidebar.title("🍔 Food Delivery Analysis")
page = st.sidebar.radio(
    "Select Analysis",
    ["📊 Dashboard Overview", "🔍 Exploratory Data Analysis", "🎯 Churn Prediction", 
     "⏱️ Delivery Delay Prediction", "👥 Customer Segmentation", "🔮 Predict Customer Churn"]
)

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================
if page == "📊 Dashboard Overview":
    st.title("📊 Food Delivery App Analysis Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(customer_df):,}")
    with col2:
        st.metric("Total Orders", f"{len(order_df):,}")
    with col3:
        churn_rate = (customer_df['churned'] == 'Inactive').sum() / len(customer_df) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        avg_spent = customer_df['total_spent'].mean()
        st.metric("Avg Spend/Customer", f"${avg_spent:,.0f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = customer_df['churned'].value_counts()
        fig, ax = plt.subplots()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Customer Churn Status')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top 10 Cities by Order Volume")
        city_orders = order_df['city'].value_counts().head(10)
        fig, ax = plt.subplots()
        city_orders.plot(kind='barh', ax=ax, color='#3498db')
        ax.set_xlabel('Number of Orders')
        ax.set_title('Top 10 Cities by Order Volume')
        st.pyplot(fig)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Total Spent")
        fig, ax = plt.subplots()
        ax.hist(customer_df['total_spent'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Total Spent')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Spending Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Order Frequency Distribution")
        fig, ax = plt.subplots()
        ax.hist(customer_df['order_frequency'], bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Order Frequency')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Order Frequency Distribution')
        st.pyplot(fig)

# ============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    
    st.subheader("Descriptive Statistics - Customer Level Data")
    stats_df = customer_df[['total_spent', 'order_frequency', 'loyalty_points', 'days_since_last_order', 'customer_tenure_days']].describe()
    st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = customer_df['gender'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(gender_counts.index, gender_counts.values, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('Count')
        ax.set_title('Customer Distribution by Gender')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Age Group Distribution")
        age_counts = customer_df['age'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(age_counts.index, age_counts.values, color=['#9b59b6', '#f39c12', '#1abc9c', '#34495e'])
        ax.set_ylabel('Count')
        ax.set_title('Customer Distribution by Age Group')
        st.pyplot(fig)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate by City")
        churn_by_city = customer_df.groupby('city')['churned'].value_counts(normalize=True).mul(100).unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        churn_by_city.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('Percentage')
        ax.set_xlabel('City')
        ax.set_title('Churn Rate by City')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top 10 Restaurants by Order Count")
        top_restaurants = order_df['restaurant_name'].value_counts().head(10)
        fig, ax = plt.subplots()
        top_restaurants.plot(kind='barh', ax=ax, color='#16a085')
        ax.set_xlabel('Number of Orders')
        ax.set_title('Top 10 Restaurants by Order Volume')
        st.pyplot(fig)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Status Distribution")
        delivery_counts = order_df['delivery_status'].value_counts()
        fig, ax = plt.subplots()
        colors_delivery = ['#2ecc71', '#e74c3c', '#f39c12']
        ax.pie(delivery_counts.values, labels=delivery_counts.index, autopct='%1.1f%%', colors=colors_delivery)
        ax.set_title('Delivery Status Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlation Matrix")
        corr_matrix = customer_df[['total_spent', 'order_frequency', 'loyalty_points', 'days_since_last_order', 'customer_tenure_days']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Order Volume by Day of Week")
    order_df['day_of_week'] = order_df['order_date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = order_df['day_of_week'].value_counts().reindex(day_order)
    
    fig, ax = plt.subplots()
    ax.bar(day_counts.index, day_counts.values, color='#3498db')
    ax.set_ylabel('Number of Orders')
    ax.set_xlabel('Day of Week')
    ax.set_title('Order Volume by Day of Week')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# ============================================================================
# PAGE 3: CHURN PREDICTION MODEL PERFORMANCE
# ============================================================================
elif page == "🎯 Churn Prediction":
    st.title("🎯 Customer Churn Prediction Model")
    
    st.write("This model predicts which customers are likely to churn based on their behavior patterns.")
    
    # Prepare data for churn prediction
    le = LabelEncoder()
    customer_df_copy = customer_df.copy()
    customer_df_copy['churned_numeric'] = le.fit_transform(customer_df_copy['churned'])
    y = customer_df_copy['churned_numeric']
    
    features = ['days_since_last_order', 'order_frequency', 'total_spent', 'loyalty_points', 'customer_tenure_days', 'age', 'gender', 'city']
    X = customer_df_copy[features].copy()
    
    # Convert age to numeric
    X['age'] = pd.to_numeric(X['age'], errors='coerce')
    X['age'] = X['age'].fillna(X['age'].median())
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['gender', 'city'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train model
    churn_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    churn_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = churn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Training Set Size", f"{len(X_train):,}")
    with col3:
        st.metric("Test Set Size", f"{len(X_test):,}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Features Predicting Churn")
        feature_importances = pd.Series(churn_model.feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(10)
        
        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax, color='#e74c3c')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Churn Prediction Features')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Active', 'Inactive'])
        ax.set_yticklabels(['Active', 'Inactive'])
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df, use_container_width=True)

# ============================================================================
# PAGE 4: DELIVERY DELAY PREDICTION
# ============================================================================
elif page == "⏱️ Delivery Delay Prediction":
    st.title("⏱️ Delivery Delay Prediction")
    
    st.write("This model predicts whether an order will face delivery delays or cancellations.")
    
    # Prepare data
    y = order_df['delivery_delay']
    features = ['restaurant_name', 'city', 'category', 'price', 'quantity', 'order_hour', 'order_day_of_week']
    X = order_df[features].copy()
    
    # One-hot encode
    X = pd.get_dummies(X, columns=['restaurant_name', 'city', 'category'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train model
    delay_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    delay_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = delay_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Training Set Size", f"{len(X_train):,}")
    with col3:
        st.metric("Test Set Size", f"{len(X_test):,}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Features Predicting Delays")
        feature_importances = pd.Series(delay_model.feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(10)
        
        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax, color='#f39c12')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Delivery Delay Prediction Features')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['On-Time', 'Delayed'])
        ax.set_yticklabels(['On-Time', 'Delayed'])
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Top 10 Restaurants by Delay Rate")
    delay_rate = order_df.groupby('restaurant_name')['delivery_delay'].mean().nlargest(10).sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    delay_rate.plot(kind='barh', ax=ax, color='#e67e22')
    ax.set_xlabel('Delay Rate (Proportion)')
    ax.set_title('Top 10 Restaurants by Delivery Delay Rate')
    st.pyplot(fig)

# ============================================================================
# PAGE 5: CUSTOMER SEGMENTATION
# ============================================================================
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Analysis")
    
    st.write("This analysis segments customers into 3 groups based on their purchasing behavior using K-Means clustering.")
    
    # Prepare data for segmentation
    customer_df_seg = customer_df.copy()
    customer_df_seg['age'] = pd.to_numeric(customer_df_seg['age'], errors='coerce')
    customer_df_seg['age'] = customer_df_seg['age'].fillna(customer_df_seg['age'].median())
    
    # Select features
    segmentation_features = ['days_since_last_order', 'order_frequency', 'total_spent']
    X = customer_df_seg[segmentation_features]
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df_seg['Segment'] = kmeans.fit_predict(X_scaled)
    
    # Characterize segments
    segment_analysis = customer_df_seg.groupby('Segment')[segmentation_features].mean().reset_index()
    
    segment_names = {}
    vip_segment = segment_analysis.sort_values(by='days_since_last_order', ascending=True).iloc[0]['Segment']
    segment_names[vip_segment] = 'Loyal & Recent (VIP)'
    
    at_risk_segment = segment_analysis.sort_values(by='days_since_last_order', ascending=False).iloc[0]['Segment']
    if at_risk_segment != vip_segment:
        segment_names[at_risk_segment] = 'At-Risk (Churn)'
    
    remaining_segment = [i for i in range(3) if i not in segment_names][0]
    segment_names[remaining_segment] = 'New/Low-Value'
    
    customer_df_seg['Segment_Name'] = customer_df_seg['Segment'].map(segment_names)
    
    # Display segment statistics
    st.subheader("Segment Characteristics")
    segment_stats = customer_df_seg.groupby('Segment_Name').agg({
        'customer_id': 'count',
        'total_spent': 'mean',
        'order_frequency': 'mean',
        'days_since_last_order': 'mean',
        'loyalty_points': 'mean'
    }).round(2)
    segment_stats.columns = ['Customer Count', 'Avg Spent', 'Avg Order Frequency', 'Avg Days Since Last Order', 'Avg Loyalty Points']
    st.dataframe(segment_stats, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segmentation (RFM Analysis)")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_map = {'Loyal & Recent (VIP)': '#2ecc71', 'At-Risk (Churn)': '#e74c3c', 'New/Low-Value': '#3498db'}
        for segment in customer_df_seg['Segment_Name'].unique():
            segment_data = customer_df_seg[customer_df_seg['Segment_Name'] == segment]
            ax.scatter(segment_data['days_since_last_order'], segment_data['total_spent'], 
                      label=segment, s=50, alpha=0.6, color=colors_map.get(segment, '#95a5a6'))
        ax.set_xlabel('Recency (Days Since Last Order - Lower is Better)')
        ax.set_ylabel('Total Spent (Monetary Value)')
        ax.set_title('Customer Segmentation (RFM Analysis)')
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Segment Distribution")
        segment_counts = customer_df_seg['Segment_Name'].value_counts()
        fig, ax = plt.subplots()
        colors_pie = ['#2ecc71', '#e74c3c', '#3498db']
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors_pie)
        ax.set_title('Customer Distribution by Segment')
        st.pyplot(fig)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment: Order Frequency vs Loyalty Points")
        fig, ax = plt.subplots()
        for segment in customer_df_seg['Segment_Name'].unique():
            segment_data = customer_df_seg[customer_df_seg['Segment_Name'] == segment]
            ax.scatter(segment_data['order_frequency'], segment_data['loyalty_points'], 
                      label=segment, s=50, alpha=0.6, color=colors_map.get(segment, '#95a5a6'))
        ax.set_xlabel('Order Frequency')
        ax.set_ylabel('Loyalty Points')
        ax.set_title('Order Frequency vs Loyalty Points by Segment')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Churn Rate by Segment")
        churn_by_segment = customer_df_seg.groupby('Segment_Name')['churned'].apply(lambda x: (x == 'Inactive').sum() / len(x) * 100)
        fig, ax = plt.subplots()
        churn_by_segment.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('Churn Rate (%)')
        ax.set_xlabel('Segment')
        ax.set_title('Churn Rate by Customer Segment')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# ============================================================================
# PAGE 6: PREDICT CUSTOMER CHURN (Interactive Prediction)
# ============================================================================
elif page == "🔮 Predict Customer Churn":
    st.title("🔮 Predict Customer Churn")
    
    st.write("Enter customer details to predict whether they are likely to churn.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
        order_frequency = st.number_input("Order Frequency", min_value=0, max_value=100, value=10)
        total_spent = st.number_input("Total Spent ($)", min_value=0.0, max_value=10000.0, value=500.0)
        loyalty_points = st.number_input("Loyalty Points", min_value=0, max_value=1000, value=100)
    
    with col2:
        customer_tenure_days = st.number_input("Customer Tenure (Days)", min_value=0, max_value=1000, value=365)
        age = st.selectbox("Age Group", ["Teenager", "Adult", "Senior"])
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        city = st.selectbox("City", ["Islamabad", "Karachi", "Lahore", "Multan", "Peshawar"])
    
    if st.button("🔮 Predict Churn Status", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'days_since_last_order': [days_since_last_order],
            'order_frequency': [order_frequency],
            'total_spent': [total_spent],
            'loyalty_points': [loyalty_points],
            'customer_tenure_days': [customer_tenure_days],
            'age': [age],
            'gender': [gender],
            'city': [city]
        })
        
        # Convert age to numeric
        age_mapping = {"Teenager": 15, "Adult": 35, "Senior": 65}
        input_data['age'] = input_data['age'].map(age_mapping)
        
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=['gender', 'city'], drop_first=True)
        
        # Ensure all feature columns exist
        for col in feature_info['feature_names']:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Select only the required features in the correct order
        input_data = input_data[feature_info['feature_names']]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.divider()
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_status = feature_info['label_encoder_classes'][prediction]
            if churn_status == "Active":
                st.success(f"✅ **Prediction: {churn_status}**")
                st.write("This customer is likely to remain active.")
            else:
                st.warning(f"⚠️ **Prediction: {churn_status}**")
                st.write("This customer is at risk of churning. Consider retention strategies.")
        
        with col2:
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots()
            classes = feature_info['label_encoder_classes']
            ax.bar(classes, prediction_proba, color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel('Probability')
            ax.set_title('Churn Prediction Confidence')
            ax.set_ylim([0, 1])
            for i, v in enumerate(prediction_proba):
                ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)

st.sidebar.divider()
st.sidebar.info("📊 Food Delivery App Analysis Dashboard v2.0")
