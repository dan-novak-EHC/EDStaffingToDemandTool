import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.data_processing import load_and_validate_data, prepare_hourly_data
from utils.forecasting import forecast_demand
from utils.visualization import (
    plot_time_series,
    plot_heatmap,
    plot_day_of_week_pattern,
    plot_hourly_pattern,
    plot_monthly_pattern,
    plot_forecast_comparison,
    plot_staffing_vs_demand
)
from utils.optimization import optimize_staffing, calculate_staffing_metrics
import os
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'true'

# Page configuration
st.set_page_config(
    page_title="ED Staffing Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Main header
st.markdown('<div class="main-header">🏥 Emergency Department Staffing Optimizer</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=ED+Staffing", use_container_width=True)
    st.markdown("### 📊 Navigation")
    st.markdown("Use the tabs below to:")
    st.markdown("1. **Upload** historical data")
    st.markdown("2. **Analyze** patterns")
    st.markdown("3. **Forecast** future demand")
    st.markdown("4. **Optimize** staffing")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("This tool helps Emergency Department teams optimize staffing based on historical patient arrivals and forecasted demand.")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Support:** Qualified Health Assistant")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Upload", "📊 Historical Analysis", "🔮 Demand Forecast", "👥 Staffing Optimization"])

# ============================================================================
# TAB 1: DATA UPLOAD
# ============================================================================
with tab1:
    st.markdown('<div class="sub-header">Upload Historical Patient Arrival Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>📋 Data Requirements:</strong>
        <ul>
            <li>File format: Excel (.xlsx)</li>
            <li>Can be patient-level data (one row per patient) or pre-aggregated hourly data</li>
            <li>Must have a datetime column (e.g., "Arrival Instant", "Arrival Date")</li>
            <li>Recommended: At least 1 year of data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Option 1: Upload file (for smaller files)
        st.markdown("#### 📤 Option 1: Upload File")
        uploaded_file = st.file_uploader(
            "Choose an Excel file (recommended for files < 20 MB)",
            type=['xlsx'],
            help="Upload your historical patient arrival data in Excel format",
            key="file_uploader"
        )
        
        st.markdown("---")
        
        # Option 2: Select from workspace (for larger files)
        st.markdown("#### 📁 Option 2: Select from Workspace")
        st.info("💡 **For large files (>20 MB):** Upload your Excel file to the workspace using the file explorer (left sidebar), then select it below.")
        
        import os
        import glob
        
        # Find all Excel files in the workspace
        excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
        excel_files = [f for f in excel_files if not f.startswith('~')]  # Exclude temp files
        
        selected_file = None
        if excel_files:
            selected_file_name = st.selectbox(
                "Select an Excel file from workspace",
                options=["-- Select a file --"] + sorted(excel_files),
                help="Excel files found in the workspace directory"
            )
            
            if selected_file_name != "-- Select a file --":
                selected_file = selected_file_name
                st.success(f"✅ Selected: {selected_file_name}")
        else:
            st.warning("⚠️ No Excel files found in workspace.")
            with st.expander("📖 How to upload files to workspace"):
                st.markdown("""
                1. Look at the **left sidebar** in VS Code
                2. Find the **Explorer** panel (file icon)
                3. Right-click in the file list
                4. Select **"Upload..."**
                5. Choose your Excel file
                6. Refresh this page to see it in the dropdown
                """)
        
        # Determine which file to process
        file_to_process = None
        file_source = None
        
        if uploaded_file is not None:
            file_to_process = uploaded_file
            file_source = f"Uploaded: {uploaded_file.name}"
        elif selected_file is not None:
            file_to_process = selected_file
            file_source = f"Workspace: {selected_file}"
        
        if file_to_process is not None:
            with st.spinner("Loading and validating data..."):
                try:
                    # Load and validate data
                    df, validation_results = load_and_validate_data(file_to_process)
                    
                    if validation_results['is_valid']:
                        st.session_state.data = df
                        
                        st.markdown(f'<div class="success-box">✅ <strong>Data loaded successfully!</strong><br>Source: {file_source}</div>', unsafe_allow_html=True)
                        
                        # Display data summary
                        st.markdown("#### 📈 Data Summary")
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Total Records", f"{len(df):,}")
                        with col_b:
                            st.metric("Date Range", f"{(df['datetime'].max() - df['datetime'].min()).days} days")
                        with col_c:
                            st.metric("Total Arrivals", f"{df['arrivals'].sum():,.0f}")
                        with col_d:
                            st.metric("Avg Daily Arrivals", f"{df.groupby(df['datetime'].dt.date)['arrivals'].sum().mean():.1f}")
                        
                        # Show if patient-level data was aggregated
                        if validation_results.get('is_patient_level', False):
                            st.info(f"ℹ️ Detected and aggregated patient-level data: {validation_results.get('original_rows', 0):,} patient records → {len(df):,} hourly records")
                        
                        # Show sample data
                        st.markdown("#### 🔍 Data Preview")
                        st.dataframe(df.head(24), use_container_width=True)
                        
                        # Validation details
                        with st.expander("📋 Validation Details"):
                            for warning in validation_results.get('warnings', []):
                                st.warning(warning)
                            st.json({k: v for k, v in validation_results.items() if k not in ['warnings', 'issues']})
                    
                    else:
                        st.error("❌ Data validation failed. Please check the following issues:")
                        for issue in validation_results.get('issues', []):
                            st.error(f"⚠️ {issue}")
                
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.markdown("#### 💡 Tips")
        st.markdown("""
        **Patient-Level Data:**
        - One row per patient visit
        - Must have arrival datetime column
        - App will automatically aggregate to hourly
        
        **Pre-Aggregated Data:**
        - One row per hour
        - Datetime + arrival count columns
        
        **For Large Files:**
        - Use workspace upload method
        - Avoids browser upload limits
        """)
        
        if st.session_state.data is not None:
            st.markdown("---")
            st.markdown("#### ✅ Ready for Analysis")
            st.success(f"Data loaded: {len(st.session_state.data):,} records")
            st.info("👉 Go to **Historical Analysis** tab")

# ============================================================================
# TAB 2: HISTORICAL ANALYSIS
# ============================================================================
with tab2:
    st.markdown('<div class="sub-header">Historical Patient Arrival Patterns</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the **Data Upload** tab first.")
    else:
        df = st.session_state.data
        
        # Time series plot
        st.markdown("#### 📈 Time Series - Daily Arrivals")
        fig_ts = plot_time_series(df)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Heatmap
        st.markdown("#### 🔥 Arrival Heatmap - Day of Week × Hour of Day")
        fig_heatmap = plot_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Pattern analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Day of Week Pattern")
            fig_dow = plot_day_of_week_pattern(df)
            st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            st.markdown("#### 🕐 Hourly Pattern")
            fig_hourly = plot_hourly_pattern(df)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Monthly pattern
        st.markdown("#### 📆 Monthly Pattern")
        fig_monthly = plot_monthly_pattern(df)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Statistical summary
        with st.expander("📊 Statistical Summary by Hour of Day"):
            hourly_stats = df.groupby('hour')['arrivals'].agg([
                ('Mean', 'mean'),
                ('Median', 'median'),
                ('Std Dev', 'std'),
                ('Min', 'min'),
                ('Max', 'max'),
                ('75th Percentile', lambda x: x.quantile(0.75)),
                ('90th Percentile', lambda x: x.quantile(0.90))
            ]).round(2)
            st.dataframe(hourly_stats, use_container_width=True)

# ============================================================================
# TAB 3: DEMAND FORECAST
# ============================================================================
with tab3:
    st.markdown('<div class="sub-header">Forecast Future Patient Demand</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the **Data Upload** tab first.")
    else:
        df = st.session_state.data
        
        # Forecast parameters
        st.markdown("#### ⚙️ Forecast Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_growth = st.number_input(
                "Annual Growth Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Expected annual growth in patient volume"
            )
        
        with col2:
            percentile = st.number_input(
                "Demand Percentile (%)",
                min_value=50,
                max_value=99,
                value=75,
                step=5,
                help="Percentile of historical demand to use for forecasting"
            )
        
        with col3:
            forecast_year = st.number_input(
                "Forecast Year",
                min_value=2024,
                max_value=2030,
                value=2026,
                step=1,
                help="Year to forecast"
            )
        
        with col4:
            include_seasonality = st.checkbox(
                "Include Seasonality",
                value=True,
                help="Apply monthly seasonality factors"
            )
        
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                try:
                    forecast_df = forecast_demand(
                        df,
                        forecast_year=forecast_year,
                        annual_growth_rate=annual_growth / 100,
                        percentile=percentile / 100,
                        include_seasonality=include_seasonality
                    )
                    
                    st.session_state.forecast_data = forecast_df
                    
                    st.markdown('<div class="success-box">✅ <strong>Forecast generated successfully!</strong></div>', unsafe_allow_html=True)
                    
                    # Forecast summary
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Forecast Year", forecast_year)
                    with col_b:
                        st.metric("Total Forecasted Arrivals", f"{forecast_df['forecasted_arrivals'].sum():,.0f}")
                    with col_c:
                        historical_total = df['arrivals'].sum() * (365 / ((df['datetime'].max() - df['datetime'].min()).days))
                        growth_pct = ((forecast_df['forecasted_arrivals'].sum() / historical_total) - 1) * 100
                        st.metric("Growth vs Historical", f"{growth_pct:.1f}%")
                    with col_d:
                        st.metric("Avg Daily Arrivals", f"{forecast_df.groupby(forecast_df['datetime'].dt.date)['forecasted_arrivals'].sum().mean():.1f}")
                
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
        
        # Display forecast results
        if st.session_state.forecast_data is not None:
            forecast_df = st.session_state.forecast_data
            
            st.markdown("---")
            st.markdown("#### 📊 Forecast Results")
            
            # Comparison plot
            fig_forecast = plot_forecast_comparison(df, forecast_df)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Monthly breakdown
            st.markdown("#### 📅 Monthly Forecast Breakdown")
            monthly_forecast = forecast_df.groupby(forecast_df['datetime'].dt.to_period('M')).agg({
                'forecasted_arrivals': 'sum',
                'baseline_arrivals': 'sum'
            }).round(0)
            monthly_forecast.columns = ['Forecasted Arrivals', 'Baseline (Historical Pattern)']
            monthly_forecast['Growth'] = ((monthly_forecast['Forecasted Arrivals'] / monthly_forecast['Baseline (Historical Pattern)']) - 1) * 100
            monthly_forecast['Growth'] = monthly_forecast['Growth'].round(1).astype(str) + '%'
            st.dataframe(monthly_forecast, use_container_width=True)
            
            # Download forecast
            st.markdown("#### 💾 Download Forecast Data")
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"ed_forecast_{forecast_year}.csv",
                mime="text/csv"
            )

# ============================================================================
# TAB 4: STAFFING OPTIMIZATION
# ============================================================================
with tab4:
    st.markdown('<div class="sub-header">Optimize Staffing Schedule</div>', unsafe_allow_html=True)
    
    if st.session_state.forecast_data is None:
        st.warning("⚠️ Please generate a forecast in the **Demand Forecast** tab first.")
    else:
        forecast_df = st.session_state.forecast_data
        
        # Staffing parameters
        st.markdown("#### ⚙️ Staffing Parameters")
        
        # Provider types
        st.markdown("##### 👨‍⚕️ Provider Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**MD (Physician)**")
            md_shift_length = st.number_input("MD Shift Length (hours)", min_value=4, max_value=12, value=8, step=1, key="md_shift")
            md_first_hour = st.number_input("MD - Patients in First Hour", min_value=1, max_value=10, value=3, step=1, key="md_first")
            md_middle_hours = st.number_input("MD - Patients per Middle Hour", min_value=1, max_value=10, value=2, step=1, key="md_middle")
            md_last_hour = st.number_input("MD - Patients in Last Hour", min_value=0, max_value=10, value=1, step=1, key="md_last")
            md_cost = st.number_input("MD - Cost per Hour ($)", min_value=0, max_value=500, value=200, step=10, key="md_cost")
        
        with col2:
            st.markdown("**APP (Advanced Practice Provider)**")
            app_shift_length = st.number_input("APP Shift Length (hours)", min_value=4, max_value=12, value=10, step=1, key="app_shift")
            app_first_hour = st.number_input("APP - Patients in First Hour", min_value=1, max_value=10, value=3, step=1, key="app_first")
            app_middle_hours = st.number_input("APP - Patients per Middle Hour", min_value=1, max_value=10, value=2, step=1, key="app_middle")
            app_last_hour = st.number_input("APP - Patients in Last Hour", min_value=0, max_value=10, value=1, step=1, key="app_last")
            app_cost = st.number_input("APP - Cost per Hour ($)", min_value=0, max_value=500, value=120, step=10, key="app_cost")
        
        # Constraints
        st.markdown("##### 🔒 Staffing Constraints")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            min_md = st.number_input("Minimum MDs per Hour", min_value=0, max_value=10, value=1, step=1)
            max_md = st.number_input("Maximum MDs per Hour", min_value=1, max_value=20, value=10, step=1)
        
        with col4:
            min_app = st.number_input("Minimum APPs per Hour", min_value=0, max_value=10, value=1, step=1)
            max_app = st.number_input("Maximum APPs per Hour", min_value=1, max_value=20, value=10, step=1)
        
        with col5:
            optimization_goal = st.selectbox(
                "Optimization Goal",
                ["Minimize RMSE", "Minimize Total Hours", "Minimize Cost", "Minimize Understaffing"],
                index=0,
                help="What should the optimizer prioritize?"
            )
        
        # Time period selection
        st.markdown("##### 📅 Optimization Period")
        col6, col7 = st.columns(2)
        
        with col6:
            optimize_full_year = st.checkbox("Optimize Full Year", value=False, help="Optimize all 12 months (slower)")
        
        with col7:
            if not optimize_full_year:
                selected_month = st.selectbox(
                    "Select Month to Optimize",
                    range(1, 13),
                    format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
                    index=0
                )
        
        # Build provider types configuration
        provider_types = {
            'MD': {
                'shift_length': md_shift_length,
                'productivity': [md_first_hour] + [md_middle_hours] * (md_shift_length - 2) + [md_last_hour],
                'cost_per_hour': md_cost,
                'min_per_hour': min_md,
                'max_per_hour': max_md
            },
            'APP': {
                'shift_length': app_shift_length,
                'productivity': [app_first_hour] + [app_middle_hours] * (app_shift_length - 2) + [app_last_hour],
                'cost_per_hour': app_cost,
                'min_per_hour': min_app,
                'max_per_hour': max_app
            }
        }
        
        # Optimize button
        if st.button("🎯 Optimize Staffing", type="primary"):
            with st.spinner("Optimizing staffing schedule... This may take a minute."):
                try:
                    # Filter data for selected period
                    if optimize_full_year:
                        optimization_data = forecast_df.copy()
                        period_label = f"Full Year {forecast_df['datetime'].dt.year.iloc[0]}"
                    else:
                        optimization_data = forecast_df[forecast_df['datetime'].dt.month == selected_month].copy()
                        period_label = datetime(2000, selected_month, 1).strftime('%B %Y')
                        period_label = period_label.replace('2000', str(forecast_df['datetime'].dt.year.iloc[0]))
                    
                    # Run optimization
                    results = optimize_staffing(
                        demand_data=optimization_data,
                        provider_types=provider_types,
                        optimization_goal=optimization_goal.lower().replace(' ', '_')
                    )
                    
                    st.session_state.optimization_results = results
                    
                    st.markdown('<div class="success-box">✅ <strong>Optimization complete!</strong></div>', unsafe_allow_html=True)
                    
                    # Display metrics
                    st.markdown(f"#### 📊 Optimization Results - {period_label}")
                    
                    metrics = results['metrics']
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    
                    with col_a:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with col_b:
                        st.metric("Total Staff Hours", f"{metrics['total_staff_hours']:,.0f}")
                    with col_c:
                        st.metric("Total Cost", f"${metrics['total_cost']:,.0f}")
                    with col_d:
                        st.metric("Coverage Rate", f"{metrics['coverage_rate']:.1f}%")
                    with col_e:
                        st.metric("Avg Understaffing", f"{metrics['avg_understaffing']:.2f}")
                    
                    # Staffing vs Demand visualization
                    st.markdown("#### 📈 Staffing vs Demand")
                    fig_staffing = plot_staffing_vs_demand(results)
                    st.plotly_chart(fig_staffing, use_container_width=True)
                    
                    # Schedule summary
                    st.markdown("#### 📋 Recommended Schedule")
                    
                    schedule_df = results['schedule']
                    
                    # Add day of week and date for readability
                    schedule_df['Day of Week'] = schedule_df['datetime'].dt.day_name()
                    schedule_df['Date'] = schedule_df['datetime'].dt.date
                    schedule_df['Hour'] = schedule_df['datetime'].dt.hour
                    
                    # Reorder columns
                    display_cols = ['Date', 'Day of Week', 'Hour', 'demand', 'total_capacity', 
                                    'patients_seen', 'unmet_demand', 'MD', 'APP']
                    schedule_display = schedule_df[display_cols].copy()
                    schedule_display.columns = ['Date', 'Day', 'Hour', 'Demand', 'Capacity', 
                                                'Seen', 'Unmet', 'MDs', 'APPs']
                    
                    st.dataframe(schedule_display, use_container_width=True, height=400)
                    
                    # Provider summary
                    st.markdown("#### 👥 Provider Summary")
                    
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        st.markdown("**MD Schedule**")
                        md_summary = schedule_df.groupby(['Date', 'Day of Week'])['MD'].max().reset_index()
                        md_summary.columns = ['Date', 'Day of Week', 'Max MDs on Duty']
                        st.dataframe(md_summary, use_container_width=True)
                    
                    with col_y:
                        st.markdown("**APP Schedule**")
                        app_summary = schedule_df.groupby(['Date', 'Day of Week'])['APP'].max().reset_index()
                        app_summary.columns = ['Date', 'Day of Week', 'Max APPs on Duty']
                        st.dataframe(app_summary, use_container_width=True)
                    
                    # Download results
                    st.markdown("#### 💾 Download Results")
                    
                    # Prepare Excel download
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        schedule_display.to_excel(writer, sheet_name='Schedule', index=False)
                        md_summary.to_excel(writer, sheet_name='MD Summary', index=False)
                        app_summary.to_excel(writer, sheet_name='APP Summary', index=False)
                        
                        # Metrics sheet
                        metrics_df = pd.DataFrame([metrics])
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Schedule as Excel",
                        data=excel_data,
                        file_name=f"ed_staffing_schedule_{period_label.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <p><strong>ED Staffing Optimizer</strong> | Powered by Qualified Health Assistant</p>
    <p>For support, use the chat button in the lower right corner</p>
</div>
""", unsafe_allow_html=True)
