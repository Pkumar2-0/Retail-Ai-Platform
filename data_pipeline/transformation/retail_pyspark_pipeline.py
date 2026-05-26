from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofmonth, month, to_date, weekofyear, year


spark = SparkSession.builder.appName("retail-raw-staged-curated-pipeline").getOrCreate()


RAW_PATH = "Files/retail/raw/Walmart.csv"
STAGED_PATH = "Files/retail/staged/walmart_staged"
CURATED_PATH = "Files/retail/curated/walmart_curated"
STORE_SUMMARY_PATH = "Files/retail/curated/store_sales_summary"
MONTHLY_SUMMARY_PATH = "Files/retail/curated/monthly_sales_summary"


raw_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(RAW_PATH)
)

staged_df = (
    raw_df
    .dropDuplicates()
    .withColumn("Date", to_date(col("Date"), "dd-MM-yyyy"))
    .filter(col("Date").isNotNull())
    .filter(col("Weekly_Sales").isNotNull())
)

staged_df.write.mode("overwrite").parquet(STAGED_PATH)

curated_df = (
    staged_df
    .withColumn("Year", year(col("Date")))
    .withColumn("Month", month(col("Date")))
    .withColumn("Day", dayofmonth(col("Date")))
    .withColumn("Week", weekofyear(col("Date")))
)

curated_df.write.mode("overwrite").parquet(CURATED_PATH)

curated_df.createOrReplaceTempView("retail_sales")

store_summary = spark.sql(
    """
    SELECT
        Store,
        COUNT(*) AS record_count,
        SUM(Weekly_Sales) AS total_sales,
        AVG(Weekly_Sales) AS avg_weekly_sales
    FROM retail_sales
    GROUP BY Store
    ORDER BY total_sales DESC
    """
)
store_summary.write.mode("overwrite").parquet(STORE_SUMMARY_PATH)

monthly_summary = spark.sql(
    """
    SELECT
        Year,
        Month,
        SUM(Weekly_Sales) AS total_sales,
        AVG(Weekly_Sales) AS avg_weekly_sales,
        AVG(Fuel_Price) AS avg_fuel_price,
        AVG(CPI) AS avg_cpi,
        AVG(Unemployment) AS avg_unemployment
    FROM retail_sales
    GROUP BY Year, Month
    ORDER BY Year, Month
    """
)
monthly_summary.write.mode("overwrite").parquet(MONTHLY_SUMMARY_PATH)

print("PySpark retail data pipeline completed.")
print(f"Raw rows: {raw_df.count()}")
print(f"Staged rows: {staged_df.count()}")
print(f"Curated rows: {curated_df.count()}")
