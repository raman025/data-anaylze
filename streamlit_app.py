import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and modern styling
st.markdown("""
<style>
    /* Beautiful gradient background with animation */
    .main {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass morphism effect for main content */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin: 20px;
        padding: 30px;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced header with glow effect */
    .main-header {
        font-size: 4rem;
        font-weight: bold;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 400% 400%;
        animation: gradientFlow 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 40px rgba(255, 255, 255, 0.3);
        filter: drop-shadow(0 0 25px rgba(255, 255, 255, 0.2));
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideInLeft 0.8s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Glass morphism cards with hover effects */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        color: white !important;
        animation: fadeInScale 0.6s ease-out;
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    /* Enhanced buttons with animations */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border: none;
        border-radius: 30px;
        color: white !important;
        font-weight: bold;
        padding: 12px 30px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
        animation: buttonPulse 2s infinite;
    }
    
    @keyframes buttonPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* Enhanced tabs with animations */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 8px;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        color: white !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out 0.4s both;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.3);
        color: white !important;
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    /* Enhanced metrics with glow */
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out 0.6s both;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    /* Enhanced file uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        padding: 2rem;
        text-align: center;
        color: white !important;
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out 0.8s both;
    }
    
    .stFileUploader:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Enhanced dataframes */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        animation: fadeInUp 0.8s ease-out 1s both;
    }
    
    /* Enhanced text and headers */
    .stMarkdown, .stText, .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: white !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 0.8s ease-out 1.2s both;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 0.8s ease-out 1.4s both;
    }
    
    /* Enhanced selectboxes and inputs */
    .stSelectbox, .stTextInput, .stTextArea, .stNumberInput, .stSlider, .stCheckbox {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox:hover, .stTextInput:hover, .stTextArea:hover, .stNumberInput:hover, .stSlider:hover, .stCheckbox:hover {
        border-color: rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Floating particles effect */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    /* Enhanced scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
    }
    
    /* Success and info boxes with animations */
    .success-box {
        background: rgba(76, 175, 80, 0.2);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white !important;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.2);
        animation: slideInRight 0.6s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .info-box {
        background: rgba(33, 150, 243, 0.2);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white !important;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.2);
        animation: slideInLeft 0.6s ease-out;
    }
    
    /* Enhanced plotly charts */
    .js-plotly-plot {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        animation: fadeInScale 0.8s ease-out 1.6s both;
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top: 3px solid #4ecdc4;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    n_samples = 100
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.lognormal(10.5, 0.4, n_samples),
        'purchase_amount': np.random.exponential(500, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'customer_type': np.random.choice(['New', 'Returning', 'VIP'], n_samples),
        'satisfaction_score': np.random.uniform(1, 5, n_samples),
        'is_premium': np.random.choice([True, False], n_samples)
    }
    
    # Add some missing values for demonstration
    data['income'][np.random.choice(n_samples, 10, replace=False)] = None
    data['region'][np.random.choice(n_samples, 5, replace=False)] = None
    
    df = pd.DataFrame(data)
    return df

def show_data_cleaning(df):
    """Display data cleaning options and results"""
    st.header("🧹 Data Cleaning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        missing_threshold = st.slider(
            "Missing Value Threshold (%)",
            min_value=10,
            max_value=90,
            value=50,
            help="Columns with more than this percentage of missing values will be dropped"
        )
    
    with col2:
        remove_duplicates = st.checkbox("Remove duplicates", value=True)
    
    if st.button("🚀 Clean Data", type="primary"):
        with st.spinner("🧹 Cleaning your data..."):
            # Clean the data
            cleaned_df = df.copy()
            
            # Remove duplicates
            if remove_duplicates:
                original_len = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                duplicates_removed = original_len - len(cleaned_df)
            
            # Drop columns with too many missing values
            threshold = missing_threshold / 100
            columns_to_drop = []
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() / len(cleaned_df) > threshold:
                    columns_to_drop.append(col)
            
            if columns_to_drop:
                cleaned_df = cleaned_df.drop(columns=columns_to_drop)
            
            # Fill missing values
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            
            # Store cleaned data
            st.session_state['cleaned_df'] = cleaned_df
            
            # Show results with enhanced styling
            st.success("✨ Data cleaned successfully!")
            
            # Enhanced metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Shape Change", f"{cleaned_df.shape[0]} × {cleaned_df.shape[1]}")
            with col2:
                st.metric("Columns Dropped", len(columns_to_drop))
            with col3:
                st.metric("Duplicates Removed", duplicates_removed if remove_duplicates else 0)
            with col4:
                missing_filled = df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
                st.metric("Missing Values Filled", missing_filled)
            
            # Show cleaned data preview
            st.subheader("👀 Cleaned Data Preview")
            st.dataframe(cleaned_df.head(10))
            
            # Download cleaned data
            csv_data = cleaned_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_data,
                file_name=f"cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_visualizations(df):
    """Display data visualizations"""
    st.header("📈 Data Visualizations")
    
    # Detect column types
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    st.subheader("🔍 Column Types Detected")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Numeric Columns ({})</h4>
        </div>
        """.format(len(numeric_columns)), unsafe_allow_html=True)
        for col in numeric_columns:
            st.write(f"• {col}")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🏷️ Categorical Columns ({})</h4>
        </div>
        """.format(len(categorical_columns)), unsafe_allow_html=True)
        for col in categorical_columns:
            st.write(f"• {col}")
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["Histograms", "Bar Charts", "Scatter Plots", "Correlation Heatmap"]
    )
    
    if viz_type == "Histograms" and numeric_columns:
        st.subheader("📊 Histograms")
        selected_cols = st.multiselect(
            "Select columns for histograms:",
            numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
        )
        
        if selected_cols:
            for col in selected_cols:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Charts" and categorical_columns:
        st.subheader("📊 Bar Charts")
        selected_col = st.selectbox("Select categorical column:", categorical_columns)
        
        if selected_col:
            value_counts = df[selected_col].value_counts().head(10)
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top Values in {selected_col}"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plots" and len(numeric_columns) >= 2:
        st.subheader("📊 Scatter Plots")
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Select X-axis column:", numeric_columns)
        
        with col2:
            y_col = st.selectbox("Select Y-axis column:", [col for col in numeric_columns if col != x_col])
        
        if x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap" and len(numeric_columns) >= 2:
        st.subheader("📊 Correlation Heatmap")
        corr_matrix = df[numeric_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Correlation Heatmap of Numeric Columns',
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Enhanced header with animations
    st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: white; margin-bottom: 2rem;">✨ Transform your data into insights with our powerful analysis tools</p>', unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Data Upload</h3>
            <p>Upload your CSV file to get started</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader - CSV only
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Only CSV files are supported"
        )
        
        # Sample data option with enhanced button
        if st.button("📋 Load Sample Data", type="primary"):
            with st.spinner("🎲 Generating sample data..."):
                st.session_state['df'] = create_sample_data()
                st.session_state['sample_data'] = True
            st.success("✨ Sample data loaded successfully!")
    
    # Main content area
    if uploaded_file is not None or 'df' in st.session_state:
        # Load data
        try:
            if 'sample_data' in st.session_state and st.session_state['sample_data']:
                df = st.session_state['df']
            else:
                # Check file type
                if not uploaded_file.name.lower().endswith('.csv'):
                    st.error("❌ **CSV Files Only!**")
                    st.warning("⚠️ This application only supports CSV file format.")
                    st.info("💡 Convert your file to CSV using Excel/Google Sheets → File → Save As → CSV")
                    return
                
                with st.spinner("📂 Loading your data..."):
                    df = pd.read_csv(uploaded_file)
            
            # Success message with animation
            st.success(f"🎉 Data loaded successfully! Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            
            # Create tabs with enhanced styling
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📊 Summary Analysis", "🧹 Data Cleaning", "📈 Visualizations"])
            
            with tab1:
                st.header("📋 Data Preview")
                st.dataframe(df.head(10))
                
                # Enhanced metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]:,}")
                with col3:
                    missing_total = df.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing_total:,}")
                with col4:
                    duplicates = df.duplicated().sum()
                    st.metric("Duplicates", f"{duplicates:,}")
                
                # Data types with enhanced styling
                st.subheader("📊 Data Types")
                dtype_info = df.dtypes.value_counts()
                for dtype, count in dtype_info.items():
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem; margin: 0.5rem 0;">
                        <strong>{dtype}:</strong> {count} columns
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download option
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"data_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.header("📊 Summary Analysis")
                
                # Missing values breakdown with enhanced styling
                st.subheader("❓ Missing Values Analysis")
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame([
                    {'Column': col, 'Missing_Count': count, 'Percentage': (count/len(df))*100}
                    for col, count in missing_data.items() if count > 0
                ]).sort_values('Missing_Count', ascending=False)
                
                if not missing_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Missing values by column:**")
                        st.dataframe(missing_df)
                    with col2:
                        # Create bar chart with proper column names
                        fig = px.bar(
                            missing_df, 
                            x='Column', 
                            y='Missing_Count', 
                            title="Missing Values by Column"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values found!")
                
                # Categorical analysis with enhanced styling
                categorical_columns = df.select_dtypes(include=['object']).columns
                if len(categorical_columns) > 0:
                    st.subheader("🏷️ Categorical Analysis")
                    for col in categorical_columns[:3]:
                        value_counts = df[col].value_counts().head(10)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📊 {col} (Top 10 values)</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(value_counts)
            
            with tab3:
                show_data_cleaning(df)
            
            with tab4:
                show_visualizations(df)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        # Enhanced welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div class="metric-card">
                <h2>🚀 Welcome to the Data Analysis Dashboard!</h2>
                <p style="font-size: 1.3rem; margin: 1rem 0;">
                    Upload a CSV file or load sample data to begin your data journey
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features overview with enhanced cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📋 Data Preview</h3>
                <p>Explore your data structure, types, and sample records with beautiful visualizations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Summary Analysis</h3>
                <p>Get comprehensive statistics and insights about your data quality and patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🧹 Data Cleaning</h3>
                <p>Automatically clean missing values, remove duplicates, and prepare your data for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions with enhanced styling
        st.markdown("""
        <div class="info-box">
            <h4>🎯 How to get started:</h4>
            <ol style="font-size: 1.1rem; line-height: 2;">
                <li>📁 Upload a CSV file using the sidebar</li>
                <li>🎲 Or click "Load Sample Data" to see the app in action</li>
                <li>🔍 Navigate through the tabs to explore different features</li>
                <li>📥 Download cleaned data and visualizations as needed</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
