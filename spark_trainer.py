from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class EnterpriseMLOps:
    """Scalable ML training and evaluation on distributed infrastructure."""
    
    def __init__(self):
        self.spark = SparkSession.builder.appName("EnterpriseMLOps").getOrCreate()
        self.evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    def train_with_tuning(self, data_path: str, label_col: str):
        """Train Gradient Boosted Trees with Cross-Validation."""
        data = self.spark.read.parquet(data_path)
        
        # Data Pipeline
        indexer = StringIndexer(inputCol=label_col, outputCol="label")
        assembler = VectorAssembler(inputCols=[c for c in data.columns if c != label_col], outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        
        gbt = GBTClassifier(labelCol="label", featuresCol="scaledFeatures", maxIter=20)
        pipeline = Pipeline(stages=[indexer, assembler, scaler, gbt])

        # Hyperparameter Grid
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [5, 10]) \
            .addGrid(gbt.stepSize, [0.01, 0.1]) \
            .build()

        cv = CrossValidator(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=self.evaluator,
                           numFolds=3)

        print("Starting distributed hyperparameter tuning...")
        cvModel = cv.fit(data)
        
        # Best model results
        results = cvModel.transform(data)
        auc = self.evaluator.evaluate(results)
        print(f"Training Complete. Best Model AUC: {auc:.4f}")
        
        return cvModel.bestModel

if __name__ == "__main__":
    mlops = EnterpriseMLOps()
    print("Spark MLOps trainer ready for big data workloads.")
