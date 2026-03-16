from delta.tables import DeltaTable

class DeltaManager:
    """Helper for Delta Lake table operations including vacuum and history."""
    def __init__(self, spark):
        self.spark = spark

    def upsert_data(self, df, path, merge_col):
        if not DeltaTable.isDeltaTable(self.spark, path):
            df.write.format("delta").save(path)
        else:
            delta_table = DeltaTable.forPath(self.spark, path)
            delta_table.alias("old").merge(
                df.alias("new"),
                f"old.{merge_col} = new.{merge_col}"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    def vacuum_table(self, path, hours=168):
        dt = DeltaTable.forPath(self.spark, path)
        dt.vacuum(hours)
