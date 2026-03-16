from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

class DistributedMLTrainer:
    def __init__(self, app_name="SparkMLTrainer"):
        self.spark = SparkSession.builder.appName(app_name).getOrCreate()

    def train_classifier(self, data_path, label_col="target"):
        """Train a distributed Random Forest model on S3/Parquet data."""
        df = self.spark.read.parquet(data_path)
        
        # Feature Engineering
        feature_cols = [c for c in df.columns if c != label_col]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        
        # Model Definition
        rf = RandomForestClassifier(labelCol=label_col, featuresCol="scaled_features", numTrees=100)
        
        # Pipeline Execution
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        model = pipeline.fit(df)
        
        return model

if __name__ == "__main__":
    trainer = DistributedMLTrainer()
    print("Distributed Spark ML trainer ready.")
