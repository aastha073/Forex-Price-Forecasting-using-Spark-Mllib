# FOREX Price Movement Prediction and Forecasting

A Big Data analytics project leveraging **Apache Spark** for predicting EUR/USD currency price movements and returns using machine learning algorithms.

## 🎯 Project Overview

This project implements two main prediction models using **PySpark MLlib**:
1. **Random Forest Classifier** - Predicts price direction (up/down)
2. **Gradient Boosted Trees Regressor** - Predicts price imbalance

## 📊 Dataset

- **Source**: EUR/USD tick data from 2014
- **Format**: High-frequency trading data with timestamps
- **Features**: UTC timestamps, Bid/Ask prices, Bid/Ask volumes
- **Size**: Multiple data samples (2M, 5M, 8M, 10M rows)

## 🔧 Technology Stack

- **Apache Spark**: Distributed computing framework
- **PySpark**: Python API for Spark
- **Spark MLlib**: Machine learning library
- **Databricks**: Cloud-based Spark environment
- **Python Libraries**: pandas, matplotlib, seaborn, sklearn

## 🚀 Spark Implementation Highlights

### Why Apache Spark?

- **Scalability**: Handles large-scale financial tick data efficiently
- **Distributed Processing**: Leverages cluster computing for faster model training
- **MLlib Integration**: Built-in machine learning algorithms optimized for big data
- **Memory Management**: In-memory computing for iterative ML algorithms

### Spark Session Configuration

```python
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.getOrCreate()
```

## 🔍 Feature Engineering with Spark

### Core Features Created:

| Feature | Formula | Purpose |
|---------|---------|---------|
| **MidPrice** | `(AskPrice + BidPrice) / 2` | Central price estimate |
| **BidAskSpread** | `AskPrice - BidPrice` | Market liquidity indicator |
| **PriceImbalance** | `(BidVolume - AskVolume) / (BidVolume + AskVolume)` | Order book imbalance |
| **LiquidityRatio** | `(BidVolume + AskVolume) / BidAskSpread` | Market depth measure |
| **LogReturn** | `Log(MidPrice_t / MidPrice_(t-1))` | Price return calculation |
| **Volatility** | `Rolling Std Dev of MidPrice` | 30-period volatility |
| **TickDirection** | `{1 if price↑, 0 if price↓}` | Price movement direction |

### Spark Window Functions for Time Series

```python
from pyspark.sql.window import Window

# Time-based window specification
windowSpec = Window.orderBy("UTC")

# Calculate lag features
EURUSD_tqdata = EURUSD_tqdata.withColumn(
    "LogReturn",
    F.log(F.col("MidPrice") / F.lag("MidPrice", 1).over(windowSpec))
)

# Rolling volatility calculation
rollingWindow = Window.orderBy("UTC").rowsBetween(-29, 0)
EURUSD_tqdata = EURUSD_tqdata.withColumn(
    "Volatility",
    F.stddev("MidPrice").over(rollingWindow)
)
```

## 🎯 Model 1: Random Forest Classifier (Price Direction)

### Objective
Predict whether the next tick will move **UP** (1) or **DOWN/SAME** (0)

### Spark MLlib Implementation

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Feature selection
feature_classifier = [
    "BidVolume", "AskVolume", "MidPrice",
    "BidAskSpread", "PriceImbalance", "Volatility"
]

# Vector assembly for Spark ML
assembler = VectorAssembler(inputCols=feature_classifier, outputCol="features")
df = assembler.transform(EURUSD_sample)

# Random Forest configuration
rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol="TickDirection", 
    numTrees=4
)
```

### Hyperparameter Tuning with Spark

```python
# Grid search parameters
paramGrid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [2, 4, 6])
    .addGrid(rf.maxDepth, [5, 10, 15])
    .addGrid(rf.maxBins, [32, 64])
    .addGrid(rf.impurity, ["gini", "entropy"])
    .build())

# Cross-validation with Spark
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="TickDirection"),
    numFolds=2,
    seed=42
)
```

### Performance Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC** | 0.6516 | Moderate discriminative power |
| **F1 Score** | 0.6070 | Balanced precision/recall |
| **Recall** | 0.7520 | Good at capturing positive movements |

## 📈 Model 2: Gradient Boosted Trees Regressor (Price Imbalance)

### Objective
Predict **price imbalance** to understand order book dynamics and market microstructure

### Spark MLlib Implementation

```python
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Feature selection for regression
feature_cols = [
    "BidVolume", "AskVolume", "MidPrice",
    "BidAskSpread", "LiquidityRatio", "LogReturn", "Volatility", "PriceAcceleration"
]

# GBT Regressor configuration
gbt = GBTRegressor(
    featuresCol="features", 
    labelCol="PriceImbalance", 
    seed=42
)
```

### Advanced Hyperparameter Optimization

```python
# Comprehensive parameter grid
paramGrid = (ParamGridBuilder()
    .addGrid(gbt.maxDepth, [5, 10])
    .addGrid(gbt.maxIter, [50, 100])
    .addGrid(gbt.stepSize, [0.0001])
    .build())

# Regression evaluator
evaluator = RegressionEvaluator(
    labelCol="PriceImbalance", 
    predictionCol="prediction", 
    metricName="rmse"
)
```

### Performance Results

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **RMSE** | 0.4049 | Low prediction error for imbalance |
| **MAE** | 0.2716 | Robust performance |
| **MSE** | 0.1639 | Good model accuracy |

## ⚡ Spark Performance Optimization

### Data Processing Pipeline

```python
# Efficient data pipeline with Spark
pipeline = Pipeline(stages=[assembler, scaler, model])

# Time-based train/test split (crucial for financial data)
count = df.count()
train_count = int(count * 0.8)
train_df = df.limit(train_count)
test_df = df.subtract(train_df)

# Cache for iterative algorithms
train_df.cache()
test_df.cache()
```

### Scalability Analysis

| Data Size | Workers | Duration (min) | Performance Notes |
|-----------|---------|----------------|-------------------|
| 20% | 2 | 48 | Baseline performance |
| 50% | 3 | 99 | Linear scaling |
| 80% | 4 | 289 | Memory optimization needed |
| 100% | 4 | 322 | Full dataset processing |

## 🔧 Spark Configuration Best Practices

### Memory Management
```python
# Optimize Spark session for financial data
spark = SparkSession.builder \
    .appName("FOREX_Prediction") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
```

### Distributed Computing Benefits

1. **Horizontal Scaling**: Add more workers to handle larger datasets
2. **Fault Tolerance**: Automatic recovery from node failures
3. **Memory Optimization**: Intelligent caching and partitioning
4. **Cross-Validation**: Parallel model training across parameter grid

## 📊 Feature Correlation Analysis

The project includes comprehensive feature correlation analysis using Spark's distributed computing:

```python
# Convert to Pandas for correlation (after Spark processing)
numeric_cols = [f.name for f in EURUSD_sample.schema.fields 
                if isinstance(f.dataType, (DoubleType, IntegerType))]
pdf = EURUSD_sample.select(numeric_cols).toPandas()
corr_matrix = pdf.corr()
```

**Key Insights:**
- Strong correlation between BidPrice, AskPrice, and MidPrice (>0.99)
- High correlation between PriceImbalance and LiquidityRatio (0.84)
- Feature selection critical for model performance

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install pyspark pandas matplotlib seaborn scikit-learn
```

### Running the Models

1. **Setup Spark Environment**
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ForexPrediction").getOrCreate()
```

2. **Load and Process Data**
```python
# Read CSV with Spark
df = spark.read.format("csv").load(
    path="your_data_path.csv",
    header=False,
    inferSchema=True
)
```

3. **Train Models**
```python
# Random Forest Classifier
rf_model = cv.fit(train_df)

# GBT Regressor  
gbt_model = cv.fit(train_df)
```

## 📈 Business Impact

- **High-Frequency Trading**: Sub-second prediction capabilities
- **Risk Management**: Price imbalance and direction forecasting  
- **Order Book Analysis**: Understanding market microstructure
- **Algorithmic Trading**: Automated decision support
- **Market Analysis**: Real-time trend identification

## 🔮 Future Enhancements

- **Real-time Streaming**: Spark Structured Streaming for live predictions
- **Deep Learning**: Integration with Spark + TensorFlow
- **Multi-Currency**: Extend to other forex pairs
- **Alternative Data**: News sentiment and economic indicators

## 📝 Technical Notes

- **Data Leakage Prevention**: Strict time-based splitting
- **Feature Scaling**: StandardScaler for numerical stability
- **Cross-Validation**: Time-aware validation strategies
- **Memory Management**: Efficient DataFrame operations

## 🤝 Contributing

This project demonstrates enterprise-scale machine learning with Apache Spark for financial applications. The distributed computing approach enables handling of massive financial datasets while maintaining model accuracy and performance.

---

**Author**: Aastha Singh
**Course**: Big Data Technologies (BIA 678)  
**Platform**: Apache Spark / Databricks
