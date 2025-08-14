import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import warnings
import io
import base64
from datetime import datetime
import json
warnings.filterwarnings('ignore')

# Handle optional dependencies gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("📦 scipy not installed. Some statistical features will be limited.")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("📦 scikit-learn not installed. Machine learning features will be limited.")

def fix_dtypes_for_plotly(df):
    """Convert problematic numpy dtypes to standard types for Plotly compatibility"""
    df_fixed = df.copy()
    for col in df_fixed.columns:
        if str(df_fixed[col].dtype).startswith('Int') or str(df_fixed[col].dtype).startswith('Float'):
            # Convert nullable integer/float types to standard types
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').astype('float64')
        elif df_fixed[col].dtype == 'object':
            # Ensure object columns are properly handled
            df_fixed[col] = df_fixed[col].astype(str)
        elif 'datetime' in str(df_fixed[col].dtype):
            # Handle datetime columns
            df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
    return df_fixed

def safe_json_serialize(obj):
    """Safely serialize objects for JSON, handling numpy types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

# Page configuration
st.set_page_config(
    page_title="AI Data Analyzer Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Control Panel */
    .control-panel {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .control-panel h3 {
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #4facfe;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* File Uploader */
    .stFileUploader > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 2px dashed #ffffff;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div > div:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Dark mode adjustments */
    .stApp[data-theme="dark"] .metric-card {
        background: #1e1e1e;
        color: white;
    }
    
    .stApp[data-theme="dark"] .section-header {
        color: white;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'Viridis'

# Header
st.markdown("""
<div class="main-header fade-in">
    <h1>🤖 AI Data Analyzer Pro</h1>
    <p>Advanced Data Analysis with Machine Learning & Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Control Panel
with st.sidebar:
    st.markdown("""
    <div class="control-panel">
        <h3>🎛️ Control Panel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme Selection
    theme_choice = st.selectbox(
        "🎨 Choose Visualization Theme:",
        ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens']
    )
    st.session_state.theme = theme_choice
    
    # Analysis Options
    st.markdown("### 📊 Analysis Options")
    show_advanced = st.checkbox("Show Advanced Analytics", value=True)
    show_ml = st.checkbox("Enable ML Features", value=SKLEARN_AVAILABLE)
    auto_clean = st.checkbox("Auto-clean Data", value=False)
    
    # Export Options
    st.markdown("### 📤 Export Options")
    export_format = st.selectbox("Export Format:", ['CSV', 'Excel', 'JSON'])

# File Upload Section
st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your dataset to begin analysis"
)

if uploaded_file:
    # Loading animation
    with st.spinner('🔄 Processing your data...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Fix dtypes immediately after loading
            df = fix_dtypes_for_plotly(df)
            
            # Auto-clean if enabled
            if auto_clean:
                # Remove completely empty rows and columns
                df = df.dropna(how='all').dropna(axis=1, how='all')
                # Fill numeric columns with median
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                # Fill categorical columns with mode
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    if not df[col].empty:
                        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                        df[col] = df[col].fillna(mode_val)
                df = fix_dtypes_for_plotly(df)
            
            # Store in session state
            st.session_state.df = df
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

    # Dataset Overview
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">📊 Rows</h3>
            <h2 style="margin: 0; color: #333;">{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f093fb; margin-bottom: 0.5rem;">📈 Columns</h3>
            <h2 style="margin: 0; color: #333;">{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4facfe; margin-bottom: 0.5rem;">💾 Memory</h3>
            <h2 style="margin: 0; color: #333;">{memory_usage:.1f} MB</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f5576c; margin-bottom: 0.5rem;">❓ Missing</h3>
            <h2 style="margin: 0; color: #333;">{missing_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # Tabbed Interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Data Explorer", "📊 Visualizations", "🧠 ML Analytics", "📈 Statistics", "🔧 Data Cleaning"])
    
    with tab1:
        st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True, height=400)
        
        with col2:
            st.subheader("📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null': [df[col].count() for col in df.columns],
                'Null %': [round((df[col].isnull().sum() / len(df)) * 100, 1) for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, height=400)
    
    with tab2:
        st.markdown('<div class="section-header">📊 Interactive Visualizations</div>', unsafe_allow_html=True)
        
        viz_col1, viz_col2 = st.columns([1, 2])
        
        with viz_col1:
            st.subheader("🎛️ Visualization Controls")
            
            # Get column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            col_options = df.columns.tolist()
            
            plot_type = st.selectbox(
                "📈 Select Plot Type:",
                ["📊 Bar Chart", "📈 Line Chart", "🥧 Pie Chart", "📉 Histogram", 
                 "🔗 Scatter Plot", "📦 Box Plot", "🌡️ Heatmap", "📊 Distribution Plot"]
            )
        
        with viz_col2:
            if plot_type == "📊 Bar Chart":
                bar_col = st.selectbox("Select column for Bar Chart:", col_options)
                if bar_col:
                    value_counts = df[bar_col].value_counts().head(20)
                    fig = px.bar(
                        x=[str(x) for x in value_counts.index], 
                        y=[safe_json_serialize(x) for x in value_counts.values], 
                        title=f"Bar Chart of '{bar_col}'",
                        color=[safe_json_serialize(x) for x in value_counts.values],
                        color_continuous_scale=theme_choice.lower()
                    )
                    fig.update_layout(xaxis_title=bar_col, yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "🥧 Pie Chart":
                if categorical_cols:
                    cat_col = st.selectbox("Select categorical column:", categorical_cols)
                    if cat_col:
                        pie_data = df[cat_col].value_counts().head(10)
                        fig = px.pie(
                            values=[safe_json_serialize(x) for x in pie_data.values], 
                            names=[str(x) for x in pie_data.index], 
                            title=f"Distribution of '{cat_col}'"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "📈 Line Chart":
                if numeric_cols:
                    line_col = st.selectbox("Select numeric column:", numeric_cols)
                    if line_col:
                        fig = px.line(
                            x=df.index, 
                            y=df[line_col], 
                            title=f"Line Chart of '{line_col}'"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "📉 Histogram":
                if numeric_cols:
                    hist_col = st.selectbox("Select numeric column:", numeric_cols)
                    if hist_col:
                        fig = px.histogram(
                            df, 
                            x=hist_col, 
                            title=f"Distribution of '{hist_col}'",
                            color_discrete_sequence=[px.colors.qualitative.Set3[0]]
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "🔗 Scatter Plot":
                if len(numeric_cols) >= 2:
                    scatter_x = st.selectbox("Select X-axis:", numeric_cols)
                    scatter_y = st.selectbox("Select Y-axis:", numeric_cols)
                    color_col = st.selectbox("Color by (optional):", ['None'] + categorical_cols)
                    
                    if scatter_x and scatter_y:
                        if color_col != 'None':
                            fig = px.scatter(
                                df, 
                                x=scatter_x, 
                                y=scatter_y, 
                                color=color_col,
                                title=f"Scatter Plot: {scatter_x} vs {scatter_y}"
                            )
                        else:
                            fig = px.scatter(
                                df, 
                                x=scatter_x, 
                                y=scatter_y,
                                title=f"Scatter Plot: {scatter_x} vs {scatter_y}"
                            )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "📦 Box Plot":
                if numeric_cols:
                    box_col = st.selectbox("Select numeric column:", numeric_cols)
                    if box_col:
                        fig = px.box(df, y=box_col, title=f"Box Plot of '{box_col}'")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "🌡️ Heatmap":
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap",
                        color_continuous_scale=theme_choice.lower()
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">🧠 Machine Learning Analytics</div>', unsafe_allow_html=True)
        
        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ Scikit-learn is not installed. Please install it to use ML features.")
            st.code("pip install scikit-learn", language="bash")
        else:
            ml_col1, ml_col2 = st.columns(2)
            
            with ml_col1:
                st.subheader("🎯 Clustering Analysis")
                if len(numeric_cols) >= 2:
                    cluster_cols = st.multiselect("Select columns for clustering:", numeric_cols, default=numeric_cols[:2])
                    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                    
                    if st.button("🚀 Run Clustering") and cluster_cols:
                        with st.spinner("Running K-Means clustering..."):
                            # Prepare data
                            cluster_data = df[cluster_cols].dropna()
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(cluster_data)
                            
                            # Perform clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(scaled_data)
                            
                            # Visualize results
                            if len(cluster_cols) >= 2:
                                fig = px.scatter(
                                    x=cluster_data.iloc[:, 0],
                                    y=cluster_data.iloc[:, 1],
                                    color=clusters,
                                    title=f"K-Means Clustering ({n_clusters} clusters)",
                                    labels={'x': cluster_cols[0], 'y': cluster_cols[1]}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster statistics
                            cluster_df = cluster_data.copy()
                            cluster_df['Cluster'] = clusters
                            st.subheader("📊 Cluster Statistics")
                            st.dataframe(cluster_df.groupby('Cluster').mean(), use_container_width=True)
            
            with ml_col2:
                st.subheader("🔍 Anomaly Detection")
                if numeric_cols:
                    anomaly_cols = st.multiselect("Select columns for anomaly detection:", numeric_cols, default=numeric_cols[:2])
                    contamination = st.slider("Contamination rate:", 0.01, 0.5, 0.1)
                    
                    if st.button("🔍 Detect Anomalies") and anomaly_cols:
                        with st.spinner("Detecting anomalies..."):
                            # Prepare data
                            anomaly_data = df[anomaly_cols].dropna()
                            
                            # Detect anomalies
                            iso_forest = IsolationForest(contamination=contamination, random_state=42)
                            anomalies = iso_forest.fit_predict(anomaly_data)
                            
                            # Visualize results
                            if len(anomaly_cols) >= 2:
                                fig = px.scatter(
                                    x=anomaly_data.iloc[:, 0],
                                    y=anomaly_data.iloc[:, 1],
                                    color=['Normal' if x == 1 else 'Anomaly' for x in anomalies],
                                    title="Anomaly Detection Results",
                                    labels={'x': anomaly_cols[0], 'y': anomaly_cols[1]}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show anomaly statistics
                            n_anomalies = sum(anomalies == -1)
                            st.metric("🚨 Anomalies Detected", f"{n_anomalies} ({n_anomalies/len(anomalies)*100:.1f}%)")
                
                st.subheader("📈 Regression Analysis")
                if len(numeric_cols) >= 2:
                    target_col = st.selectbox("Select target variable:", numeric_cols)
                    feature_cols = st.multiselect("Select feature variables:", 
                                                [col for col in numeric_cols if col != target_col])
                    
                    if st.button("📊 Run Regression") and target_col and feature_cols:
                        with st.spinner("Running regression analysis..."):
                            # Prepare data
                            reg_data = df[feature_cols + [target_col]].dropna()
                            X = reg_data[feature_cols]
                            y = reg_data[target_col]
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train model
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            
                            # Display results
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("📊 R² Score", f"{r2:.3f}")
                            with col_b:
                                st.metric("📉 RMSE", f"{rmse:.3f}")
                            
                            # Plot predictions vs actual
                            fig = px.scatter(
                                x=y_test, 
                                y=y_pred,
                                title="Predictions vs Actual",
                                labels={'x': 'Actual', 'y': 'Predicted'}
                            )
                            fig.add_shape(
                                type="line", line=dict(dash="dash"),
                                x0=y_test.min(), y0=y_test.min(),
                                x1=y_test.max(), y1=y_test.max()
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)
        
        # Descriptive Statistics
        st.subheader("📊 Descriptive Statistics")
        if numeric_cols:
            desc_stats = df[numeric_cols].describe()
            st.dataframe(desc_stats, use_container_width=True)
        
        # Correlation Analysis
        if len(numeric_cols) >= 2:
            st.subheader("🔗 Correlation Analysis")
            corr_matrix = df[numeric_cols].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show strongest correlations
            st.subheader("💪 Strongest Correlations")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
            st.dataframe(corr_df.head(10), use_container_width=True)
        
        # Statistical Tests
        if SCIPY_AVAILABLE and len(numeric_cols) >= 2:
            st.subheader("🧪 Statistical Tests")
            
            test_col1, test_col2 = st.columns(2)
            
            with test_col1:
                st.write("**Normality Tests (Shapiro-Wilk)**")
                for col in numeric_cols[:5]:  # Limit to first 5 columns
                    sample = df[col].dropna().sample(min(5000, len(df[col].dropna())))
                    if len(sample) > 3:
                        stat, p_value = stats.shapiro(sample)
                        result = "Normal" if p_value > 0.05 else "Not Normal"
                        st.write(f"**{col}**: {result} (p={p_value:.4f})")
            
            with test_col2:
                if len(numeric_cols) >= 2:
                    st.write("**Correlation Tests (Pearson)**")
                    for i, col1 in enumerate(numeric_cols[:3]):
                        for col2 in numeric_cols[i+1:4]:
                            data1 = df[col1].dropna()
                            data2 = df[col2].dropna()
                            common_idx = data1.index.intersection(data2.index)
                            if len(common_idx) > 10:
                                corr, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                                significance = "Significant" if p_value < 0.05 else "Not Significant"
                                st.write(f"**{col1} vs {col2}**: r={corr:.3f}, {significance}")
    
    with tab5:
        st.markdown('<div class="section-header">🔧 Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
        
        # Missing Values Analysis
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() > 0:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Missing values per column:**")
                missing_df = pd.DataFrame({
                    'Column': missing_counts[missing_counts > 0].index,
                    'Missing Count': [safe_json_serialize(x) for x in missing_counts[missing_counts > 0].values],
                    'Percentage': [safe_json_serialize(round(x, 2)) for x in (missing_counts[missing_counts > 0] / len(df) * 100)]
                })
                st.dataframe(missing_df, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    missing_df, 
                    x='Column', 
                    y='Missing Count',
                    title='Missing Values by Column',
                    color='Percentage',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Cleaning Options
            st.subheader("🧹 Data Cleaning Options")
            
            cleaning_col1, cleaning_col2 = st.columns(2)
            
            with cleaning_col1:
                action = st.selectbox(
                    "Choose cleaning action:",
                    ["No Action", "Drop Rows", "Fill with Mean (Numerical)", 
                     "Fill with Mode (Categorical)", "Forward Fill", "Backward Fill"]
                )
            
            with cleaning_col2:
                if st.button("🚀 Apply Changes"):
                    with st.spinner('Processing...'):
                        if action == "Drop Rows":
                            df = df.dropna()
                            df = fix_dtypes_for_plotly(df)
                            st.session_state.df = df
                            st.success("✅ Rows with missing values dropped.")
                        elif action == "Fill with Mean (Numerical)":
                            num_cols = df.select_dtypes(include=[np.number]).columns
                            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                            df = fix_dtypes_for_plotly(df)
                            st.session_state.df = df
                            st.success("✅ Numerical columns filled with mean.")
                        elif action == "Fill with Mode (Categorical)":
                            cat_cols = df.select_dtypes(include=["object", "category"]).columns
                            for col in cat_cols:
                                if df[col].isnull().any():
                                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                                    df[col] = df[col].fillna(mode_value)
                            df = fix_dtypes_for_plotly(df)
                            st.session_state.df = df
                            st.success("✅ Categorical columns filled with mode.")
                        elif action == "Forward Fill":
                            df = df.fillna(method='ffill')
                            df = fix_dtypes_for_plotly(df)
                            st.session_state.df = df
                            st.success("✅ Missing values forward filled.")
                        elif action == "Backward Fill":
                            df = df.fillna(method='bfill')
                            df = fix_dtypes_for_plotly(df)
                            st.session_state.df = df
                            st.success("✅ Missing values backward filled.")
                        else:
                            st.info("ℹ️ No changes applied to missing values.")
                        
                        st.rerun()
        else:
            st.success("🎉 No missing values found in the dataset!")
        
        # Data Type Conversion
        st.subheader("🔄 Data Type Conversion")
        
        type_col1, type_col2 = st.columns(2)
        
        with type_col1:
            convert_col = st.selectbox("Select column to convert:", df.columns)
            new_type = st.selectbox("Select new data type:", 
                                  ["object", "int64", "float64", "datetime64", "category"])
        
        with type_col2:
            if st.button("🔄 Convert Data Type"):
                try:
                    if new_type == "datetime64":
                        df[convert_col] = pd.to_datetime(df[convert_col], errors='coerce')
                    elif new_type == "category":
                        df[convert_col] = df[convert_col].astype('category')
                    else:
                        df[convert_col] = df[convert_col].astype(new_type)
                    
                    df = fix_dtypes_for_plotly(df)
                    st.session_state.df = df
                    st.success(f"✅ Column '{convert_col}' converted to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error converting data type: {e}")
    
    # Export Section
    st.markdown('<div class="section-header">📤 Export Data</div>', unsafe_allow_html=True)
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV File",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("📈 Download Excel"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                if numeric_cols:
                    df[numeric_cols].describe().to_excel(writer, sheet_name='Statistics')
            
            st.download_button(
                label="💾 Download Excel File",
                data=buffer.getvalue(),
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with export_col3:
        if st.button("🔗 Download JSON"):
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="💾 Download JSON File",
                data=json_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;">
        <h2>🚀 Welcome to AI Data Analyzer Pro!</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">Upload your CSV or Excel file to begin advanced data analysis</p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; min-width: 200px;">
                <h4>📊 Data Exploration</h4>
                <p>Interactive data browsing and profiling</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; min-width: 200px;">
                <h4>📈 Visualizations</h4>
                <p>Dynamic charts and plots</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; min-width: 200px;">
                <h4>🧠 ML Analytics</h4>
                <p>Clustering and anomaly detection</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Installation instructions
    st.markdown('<div class="section-header">📦 Installation Requirements</div>', unsafe_allow_html=True)
    