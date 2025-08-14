# AI Data Analyzer Pro

A comprehensive data analysis web application built with Streamlit that provides interactive visualizations, statistical analysis, and machine learning capabilities for exploring and understanding your datasets.

# Features

# Data Processing
- Support for CSV and Excel file uploads
- Automated data type detection and conversion
- Missing value analysis and cleaning options
- Data profiling with completeness metrics

# Visualizations
- Interactive charts with Plotly (scatter plots, bar charts, pie charts, histograms)
- Correlation heatmaps and pair plots
- Distribution analysis with statistical overlays
- Customizable themes and color schemes

# Machine Learning
- K-Means clustering analysis
- Principal Component Analysis (PCA)
- Anomaly detection with Isolation Forest
- Linear regression modeling with performance metrics

# Advanced Analytics
- Statistical summaries and hypothesis testing
- Data quality assessment
- Automated insights generation
- Export capabilities (CSV, Excel, JSON, PDF reports)

# Installation

1. Clone the repository:
   
git clone <repository-url>
cd ai-data-analyzer


3. Install required dependencies:

pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn openpyxl xlsxwriter


4. Run the application:

streamlit run main.py


# Usage

1. Upload Data: Use the drag-and-drop interface to upload CSV or Excel files
2. Explore Data: View basic statistics, data types, and missing value analysis
3. Clean Data: Apply various cleaning operations like filling missing values or dropping rows
4. Visualize: Create interactive charts and explore data relationships
5. Analyze: Run machine learning algorithms and statistical tests
6. Export: Download processed data and analysis reports

# Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn (optional, for ML features)
- SciPy (optional, for statistical tests)

Project Structure


ai-data-analyzer/
├── main.py              # Main Streamlit application
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies


# Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

