# Distributed Spark ML Workflows â˜ï¸

Scalable data engineering and machine learning pipelines designed for enterprise-scale datasets. Built with PySpark and deployed on AWS (EMR/Glue).

## ðŸ“Š Features
- **Feature Engineering at Scale:** Handling billions of rows using Spark DataFrames.
- **MLlib Integration:** Training classification and regression models in a distributed manner.
- **AWS Integration:** S3 for data lake storage and SageMaker for model hosting.

## ðŸ§ª Sample Pipeline

### `feature_engineering.py`
```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler

def process_large_dataset(input_s3_path):
    spark = SparkSession.builder.appName("DistributedML").getOrCreate()
    df = spark.read.parquet(input_s3_path)
    
    # Example: Scale features for a credit scoring model
    assembler = VectorAssembler(inputCols=["income", "debt", "history"], outputCol="features")
    feature_df = assembler.transform(df)
    
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    model = scaler.fit(feature_df)
    return model.transform(feature_df)
```

## ðŸ”’ Security
Infrastructure as Code (IaC) following AWS Best Practices for VPC isolation and IAM least-privilege.
