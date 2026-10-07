# 🏥 Emergency Department Staffing Optimizer

A comprehensive Streamlit application for analyzing historical patient arrivals, forecasting future demand, and optimizing ED staffing schedules.

## 🎯 Features

### 📤 Data Upload & Validation
- Upload historical patient arrival data (Excel format)
- Automatic column detection and validation
- Data quality checks and reporting

### 📊 Historical Analysis
- Time series visualization with moving averages
- Heatmaps showing arrival patterns by day and hour
- Day of week and hourly pattern analysis
- Monthly seasonality visualization
- Statistical summaries

### 🔮 Demand Forecasting
- Annual growth rate application
- Percentile-based demand calculation
- Monthly seasonality factors
- Full year hourly forecasts
- Comparison with historical patterns

### 👥 Staffing Optimization
- Multi-provider type support (MD, APP, expandable)
- Configurable shift lengths and productivity patterns
- Rolling demand with carryover (realistic ED operations)
- Multiple optimization goals:
  - Minimize RMSE
  - Minimize Total Hours
  - Minimize Cost
  - Minimize Understaffing
- Constraint enforcement (min/max providers)
- Visual staffing vs. demand analysis
- Excel export of optimized schedules

## 🚀 Quick Start

### Local Installation

## 1. ## Clone the repository**
## ```bash
## git clone <your-repo-url>
## cd <your-repo-name>
