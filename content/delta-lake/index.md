+++
title="CDC with Delta Lake Streaming"
date=2022-08-07
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["big data", "lake house", "data lake"]
categories = ["scala", "spark"]
+++

{{ resize_image(path="delta-lake/images/delta-lake-logo.png", width=700, height=600, op="fit_width") }}

Change Data Capture (CDC) is a popular technique for replication of data from OLTP to OLAP data store.
Usually CDC tools integrate with transactional logs of relational databases and thus are mainly dedicated to replicate all possible data changes from relational databases. NoSQL databases are usually coming with built-in CDC for any possible data change (insert, update, delete), for example AWS DynamoDB Streams.  

In this blog-post, we will look at [Delta Lake](https://docs.delta.io/latest/index.html) table format, which supports ["merge"](https://docs.delta.io/latest/delta-update.html#upsert-into-a-table-using-merge) operation. This operation is useful when we need to update replicated data in Data Lake.

<!-- more -->

# Generate Data

Let's generate some input data and merge it using Spark streaming API. Delta Lake API comes with DSL for merging data frames into into a table.

I have prepared a Scala script which can generate CSV files with hypotetical customer orders. Every few seconds this script creates a new file which contains few hundreds of rows.

Using [Scala CLI](https://scala-cli.virtuslab.org/), run provided [csv-file-gen.scala](https://github.com/novakov-alexey/spark-elt-jobs/blob/main/scripts/csv-file-gen.scala) script. I am doing that within its cloned repository like this:

```bash
scala-cli run scripts/csv-file-gen.scala --main-class localGenerateOrders -- "data/gen/orders" 3000
```

It will start to print a number of generates rows in a new file. `3000` is pause in milliseconds after each file generation. Do not forget to stop this script afterwards, otherwise it will generate many files on your disk, but you should keep it running if you follow me and run the code below.

# Create Delta Table

Using spark-shell, or any other tools which can initiate Spark session, for example Apache Zeppelin, Jupyter, run the code below:

```scala
import io.delta.tables._
import java.io.File

val inputPath = new File("./data/gen/orders").toURI.toString
val ordersDf = spark.read
  .option("inferSchema", true)
  .option("header", true)
  .csv(inputPath)

val tablePath = new File("./data/delta/orders").toURI.toString
ordersDf.limit(0)
  .write
  .format("delta")
  .mode("overwrite")
  .save(tablePath)
```

I am using Spark **3.2.1** with Scala **2.12.15** and **Java 11.0.2**.

At this point we have a Delta table on the local file system. It is now ready for merging new changes using Spark batch or streaming queries.

In the above code, we are using already available generated files in `data/gen/orders` to create Delta Lake table itself. This is a requirement of Spark _streaming_ API to provide input schema in advance/staticaly.

# Merge streaming data

Our goal is to discover new files in the input directory and merge their content to a Delta Lake table. Essentially, we are going to run micro-batch processing, which allows to reference an intermediate DataFrame to merge its content to existing Delta Lake table.


```scala
val orders = DeltaTable.forPath(spark, tablePath) // pointing to existing table
```

We will use schema from existing `ordersDf` DataFrame to avoid manual schema definition, however you can also define required columns to be selected from intermiediate data frame for merge manually. 

```scala
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.DataFrame

val precombineKey = "last_update_time"
val primaryKey = "orderId"
val otherColumns =
    ordersDf.schema.fields
    .map(_.name)
    .filterNot(n => n == precombineKey || n == primaryKey)

// Function to upsert microBatchOutputDF into Delta table using merge
def upsertToDelta(microBatchOutputDF: DataFrame, batchId: Long) = {
  // Find the latest change for each key based on the timestamp
  // Note: For nested structs, max on struct is computed as
  // max on first struct field, if equal fall back to second fields, and so on.  
  val latestChangeForEachKey = microBatchOutputDF
    .selectExpr(
      primaryKey,
      s"struct($precombineKey, ${otherColumns.mkString(",")}) as otherCols"
    )
    .groupBy(primaryKey)
    .agg(max("otherCols").as("latest"))
    .selectExpr(primaryKey, "latest.*")

  orders.as("t")
    .merge(
      latestChangeForEachKey.as("s"),
      s"s.$primaryKey = t.$primaryKey")
    .whenMatched().updateAll()
    .whenNotMatched().insertAll()
    .execute()
}    

def runStream() = 
  spark.readStream
    .format("csv")
    .schema(ordersDf.schema)
    .load(inputPath)
    .writeStream
    .option("checkpointLocation", s"$tablePath/_checkpoints")
    .format("delta")
    .foreachBatch(upsertToDelta _)
    .outputMode("update")    
    .start()

runStream()
```    

Once `start` method is executed, Spark starts to run a streaming job, which is going to merge all incoming data based on the primareKey `orderId` and its precombine key `last_update_time`. Precombine key is used to sort all records with the same primary key and then take the record with `max(..)` value. Usually, precombine key is a time-based column which can indidicate the latest transaction happened to a specific row.

# Verify merge result

In another spark-shell terminal we are checking that there is a maximum of 1 order per each unique `orderId`. If any of the `orderId` groups show more than 1 in the count column, then merge process is not working correctly.

```scala
import io.delta.tables._
import java.io.File

val tablePath = new File("./data/delta/orders").toURI.toString
val orders = DeltaTable.forPath(spark, tablePath) // pointing to existing table
val primaryKey = "orderId"

orders.toDF
.groupBy(primaryKey)
.agg(count(primaryKey).as("count"))
.sort(desc("count"))
.show
```
Output:

```bash
+-------+-----+
|orderId|count|
+-------+-----+
|    148|    1|
|    463|    1|
|    471|    1|
|    496|    1|
|    833|    1|
|    243|    1|
|    392|    1|
|    540|    1|
|    623|    1|
|    737|    1|
|    858|    1|
|    897|    1|
|     31|    1|
|    516|    1|
|     85|    1|
|    137|    1|
|    251|    1|
|    451|    1|
|    580|    1|
|    808|    1|
+-------+-----+
only showing top 20 rows
```

Prooving that there are no duplicates for any order:

```scala
orders.toDF
res2
.groupBy(primaryKey)
.agg(count(primaryKey).as("count"))
.sort(desc("count"))
.where($"count" > 1)
.show
```

Output:

```bash
+-------+-----+
|orderId|count|
+-------+-----+
+-------+-----+
```
Result is an empty dataset as expected.


# Optimization

If we process micro batches and merge them to Delta Lake table, then sooner or later Spark will create a lot of small Parquet files inside the table folder. Some of the files will be already obsolete and will be only needed if we query historical state of the Delta Lake table. In order to optimize table reads and writes, one should compact large number of files to get smaller number files in the table folder. **Compaction** can be done via standard Spark `repartition` operation.

Compact files:

```scala
val numFiles = 4

def compact = 
  spark.read
   .format("delta")
   .load(tablePath)
   .repartition(numFiles)
   .write
   .option("dataChange", "false")
   .format("delta")
   .mode("overwrite")
   .save(tablePath)

compact
 ```

`dataChange=false` is Delta's option here to minimize potential failure to other concurrent operations on the current Delta table.

Another way to get rid of large number of files is to run Delta's **vacuum** operation, which effectively removes data files older than N number of hours.

Vacuum command deletes old files which are still part of the tables, but not used when you query the latest table state. However these files are still used if you query historical state of the table. So once you vacuum old files, you loose a possibility to query those historical data after that.

Below example removes all histoical data by setting `0` as number of hours.

 ```scala
 spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", false)
 orders.vacuum(0)
 ```

 During the compaction or vacuum you may get an exception in the running Spark streaming or batch job which can terminate your job. The reason is that both processes, i.e. main job and compact/vacuum, are trying to move table files around and thus may lead to a conflicting situations. But we remember that Delta Lake is ACID compliant, it should allow us to change data from multiple writers and still be consistent. It is still true. Delta Lake is based on [optimistic concurrency](https://docs.delta.io/latest/concurrency-control.html#id1) principles which requires clients to retry their operations upon such failures/exceptions. If you see such a failure, then make sure you repeat the same operation again or restart Spark job upon such exceptions. 

# Summary

In this blog-post we have seen that Delta Lake can easily merge new data to existing table via standard Spark API, in this case via streaming API.
Apart from the main opearion we need to run compaction and vacuum operations time by time as a separate `houskeeping` jobs to get overal better peformance when reading the table data by the main data consumers.