+++
title="Machine Learning with Flink"
date=2024-11-23
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["flink", "machine learning", "scala", "tensorflow", "pytorch"]
categories = ["scala"]
+++

If you want to add Machine Learning capabilities into your Flink job then this article is for you.
As Flink runs on Java Virtual Machine, we are constrained by the tools which JVM supports. However, there are still plenty of options to choose in order to perform model training and inference as part of a Flink job.

{{ resize_image(path="flink-ml/images/flink-ml-logo.png", width=600, height=600, op="fit_width") }}

<!-- more -->

# Supervised Machine Learning

Before we dive into Flink specifics to apply Machine Learing, let's first define key points of the Supervised Machine Learning.

In Supervised Machine Learning we feed data into the algorithm containing the right answers. 
Then we check whether the algorithm can eventually learn those answers on its own by feeding new or test data. That means we supervise the algorithm until we reach desired model performance in terms of accuracy, error rate and other metrics. 

Training phase is usually done on a batch data iteratively. In other words, our training data set is already prepared and stored on disk in one 
or multiple files. Flink can read all of those files in streaming or batch mode. 
However, it does not make sense much to use streaming mode, because a stream processing job wait for additional files on disk and won't terminate, 
if we don't setup additional configuration like streaming - bounded mode to facilitate natural job termination. The same approach would be applicable for message queue as a source, where data set would be a sequence of messages, let's say in Kafka, which we need to read all to train the model.

There is another approach when ML model is trained on the "online data", i.e. it reads data continuously and also updates model weights as part of the training. 
This process ideally never ends. Most common approach is actually training on offline data. 
In this blog post we will focus on this approach. We will use Flink batch jobs to read prepared data and run training cycle to get a trained model (state). 
Once trained model is ready, it will be stored on disk and then can be loaded by another job to do model inference as part of the business logic.

# Data Streaming and ML

Let's quickly define basic definitions for training and inference to build up further concepts on top of them:

__Model Training:__
- It is iterative process run on the finite dataset
- Training is usually done on schedule (once per hour or day or week, i.e. depends on data updates)
- If training is done on online data, then the training loop repeats infinitely and updates the internal model weights on every record or window / mini-batch

__Model Inference:__
- It is a call of a math function like "y = model(x)", where `x` is feature vector and `y` is prediction vector. 
    - In streaming mode: a call on every input record 
    - In batch mode: a call for entire mini-batch at once

Both definitions are true for Supervised and Unsupervised Machine Learning.

__Model re-training and updates__

Once model is trained and stored on disk, it may become outdated very soon. Of course, it depends whether our environment may get new 
data to learn something new from it. In many business domains, data does not change fast, so we can train a model and use it for many days or months without
re-training it again. 

In order update a model in the running Flink job, we have several options:

For cases when Flink job calls a model inside the JVM or PythonVM:
1. Restart Flink job with new model, if model loading done only on job startup.
2. Reload ML model into memory periodically or on some event. This option minimises Flink job downtime.

For cases when Flink calls a model using remote procedure call (RPC), over the network (web-service, GRPC-service, etc.): 
1. Job restart is not needed, but that external service may lead to the Flink job outage or temporary failures, when the external service is restarted.

In order to mitigate external service outage or achieve zero-time outage for the Flink side, a DevOps team can apply different techniques:
- Blue/Green deployments for the external service to serve the latest and previous model version in the same time
- Additionally, configure a network load balancer to automatically switch to a new model version once it is up and running.


# ML Libraries for Flink

As a Flink developer you might have a question - can we easily do ML training in Flink? Can it be also done with the popular ML libraries like Scikit-Learn, Pytorch, Tensorflow, etc.?

The answer to the first question is yes, we can do ML tasks in Flink. As for the second question is rather no, than yes. We can't easilly leverage Flink distributed runtime to run training process efficiently. 
Here I mean not just within Flink task, but using all Flink cluster task mmanagers. In best case, we can spread training process accross different tasks when running training phase with cross-validation (different training and test data set splits). 

If no cross validation is used, then we will be running training on a single task within a single task manager. 
Thus, this neglects the whole idea to use Flink runtime for training as it will be underutilised and will bring a lot of hassle without benefits. 
That said, the idea to use Flink without low-level integration of Flink tasks and operators with specific external library does not worth it. 
It would be the same effect as we training ML model without Flink runtime, but directly on some VMs running ML programs sequentially or in parallel for different data set splits. 

The good news is Flink has its own ML module called [Flink ML](https://nightlies.apache.org/flink/flink-ml-docs-master/). Flink ML supports training and model inference. 
It fully utilises Flink tasks to distribute training process within a Flink cluster. Of course, Flink ML module limits us by only those ML algotithms, which it supports at the moment. 
As of today, it is quite rich list of supported ML algorithms including typical data preparation algorithms, for example data normalisation. 

When it comes to other libraries, which are coming from the Python or C++ world, we can only do a model inference.
In order to load that model, we either use available JVM SDK of the target library or we can also use PyFlink and load any Python-wrapped model. 
If our ML library has C/C++ interface, then we can also use Scala/Java wrappers for those libraries, for example `libtorch`. 

# Ways to do ML in Flink

 Let's look at the table below to see all options we have to do ML tasks in Flink:

| Library Name / Approach   |  Native Training Support | Inference Support | Remark                                        |    
| ------------------------- | ----------------- | ------------------| --------------------------------------------- |
| [Flink ML](https://nightlies.apache.org/flink/flink-ml-docs-release-2.3/)| x                 | x                 | Easy to use. Most common algorithms are supported |
| [Deep Learning on Flink](https://github.com/flink-extended/dl-on-flink): Tensorflow, Pytorch integration | x  | x  | Allows to train inside a Flink cluster. Caveat: this community project requires dependency upgrades |
| JVM runtimes for model formats like PMML, ONNX  (ex: onnx-scala, flink-jpmml)|   |x   | Fast and generic way    |
| Python libraries via Flink Table API (UDF)|   |x   | Any Python library can be used, but it can be slow due to the PythonVM itself and data exchange  |
| Scala wrappers for C/C++ libraries (libtorch, etc.) |   |x   | Today is mainly limited to Pytorch or Tensorflow |
 

\* when native training with Flink runtime is not supported (empty table cell above), 
it means we can train a model outside of Flink, e.g. in any other programming language.

# Generic ML Workflow with Flink

{{ resize_image(path="flink-ml/images/ml-workflow-with-flink.png", width=1200, height=800, op="fit_width") }}

<p align="center">Figure 1. Training and Inference workflows based on Flink application</p>

## Training

In the __Training__ phase we run an application, which (1) reads data from disk, object storage or consumes a set of data from messaging system like Apache Kafka. This application can be a Flink job which uses FlinkML module or some other ML Framework via PyFlink. The ML application may (2) consume data via mini-batches or read data record by record.
While running a training loop, the application updates current model weights in memory and (3) stores them periodically to a persistent storage (file system, object storage like AWS S3). Training loop usually runs dozen of times until it reaches certain thresholds of the training metrics and some other control conditions.


## Inference

In the __Inference__ phase we run another Flink Job which may or may not share some code with the training job. 
The main point is that the inference Flink job loads already trained model and uses the same ML framework to call the model. At the beginning, inference job reads (1) data similarly to the training job, but this time it reads unseen data to (2) apply ML model and stores the results somewhere. Usually the results are stored on every record to (3) messaging system or some persistent storage.

# Flink ML Module

In this blog we uncover Flink ML module and look at its applications. In the next blog posts we look at further ways for ML tasks in Flink such as ONNX, PyFlink and C/C++ wrappers.

Flink ML module supports training and inference for the most popular supervised and unsupervised ML algorithms. 
This module uses native Flink job graph primitives to build a distributed graph of operators to perform model training or inference. 
ML job runs in distributed mode utilising all available Flink task managers in the cluster.

ML algorithms are implemented as operators. You can see them in the Flink UI when opening job graph visualisation. Flink's Table API is a basis for Flink ML. It uses Table type to represent input and output data.

{{ resize_image(path="flink-ml/images/flink-ml-job-graph.png", width=1200, height=400, op="scale") }}
<p align="center">Figure 2. Flink Job graph of LogisticRegression.</p>

Above figure demonstrates large Flink Job graph which is built by Flink Table API and Flink ML to run data preparation and Logistic Regression training process. 
We will see how to train such a model in details further.

## Algorithms supported by the FlinkML

|Task            | Algorithms                                          |
|----------------|-----------------------------------------------------|
|Classification  | K-Nearest Neighbor, Linear SVC, Logistic Regression, Naive Bayes |
|Clustering      |AgglomerativeClustering, Kmeans                                   |
|Recommendation  |Swing                                   |
|Regression      | Linear Regression |
|Statistics      | ChiSqTest |
|Evaluation      |Binary Classification, Evaluator|
|Feature Engineering | Normalizer, Scalers, Binarizer  and many others |
|Stats|ChiSqTest|


## Example: Logistic Regression

Let's implement an ML application in Flink for __Customer Churn Analysis__ by using Flink ML's Logistic Regression algorithm. This use case is quite typical task for enterprises which want to find  unhappy customers and offer better conditions to retain them.

The goal of our model is to predict whether a customer may leave a bank or not. 

As clients dataw we use syntetic data set prepared and stored in CSV file format in local filesystem.
Clients data has all required columns to train the ML model. 

## Data Preparation

Below is the data sample in CSV format which we will use by the training job:

```csv
RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1
2,15647311,Hill,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
3,15619304,Onio,502,France,Female,42,8,159660.8,3,1,0,113931.57,1
4,15701354,Boni,699,France,Female,39,1,0,2,0,0,93826.63,0
```

For our application we skip irrelevant columns such as RowNumber, CustomerId, Surname and use all other columns (features) of this dataset:

```csv
CreditScore,
Geography,
Gender,
Age,
Tenure,
Balance,
NumOfProducts,
HasCrCard,
IsActiveMember,
EstimatedSalary
```
Column `Exited` is our target label to predict. It contains binary value such as 0 or 1, which encode the following logic:
- 0 - client won't exit the bank; 
- 1 - client will exit the bank

Before we feed the selected data columns into the any ML algorithm, we need to transform this data
into numerical format. At this point we start to name data columns as features. Categorical features
should be encoded with one-hot encoder, numerical features also known as continues features should be normalised. Specific data encoding and 
normalisation is needed to achieve the highest model accuracy and the lowest error rate during the training. 

To learn more on feature engineering topic I advise you to read special literature on that. In this blog post we focus on Flink ML module itself.

### Load CSV as Flink Table

```scala
val source = FileSource
    .forRecordStreamFormat(
      TextLineInputFormat(),
      filePath
    )
    .build()

val csvStream = env
    .fromSource(source, WatermarkStrategy.noWatermarks(), "trainingData")
    .filter(l => !l.startsWith("#")) // removing comments
    .map(l =>
      // from CreditScore to Exited
      val row = l.split(",").slice(3, 14)
      Row.of(
        row(0).toDouble,
        row(1),
        row(2),
        row(3).toDouble,
        row(4).toDouble,
        row(5).toDouble,
        row(6).toDouble,
        row(7).toDouble,
        row(8).toDouble,
        row(9).toDouble,
        row(10).toDouble
      )
    )

val trainData = tEnv.fromDataStream(csvStream)    
```

`trainData` is the input data to our data preprocessing step.

### Prepare ML Features

```scala
// 1 - index Geography and Gender
val indexer = StringIndexer()
  .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
  .setInputCols("GeographyStr", "GenderStr")
  .setOutputCols("GeographyInd", "GenderInd")
```

`StringIndexer` turns string (categorical) columns into integer-indexed columns. 
For each Country and each Gender we will have an integer value from 0 to N, where N is a number of unique values in the column. 
In case of country column, N is 3 as we have 3 countries in the data set.

```scala
// 2 - OneHot Encode Geography and Gender
val geographyEncoder =
  OneHotEncoder()
    .setInputCols("GeographyInd", "GenderInd")
    .setOutputCols("Geography", "Gender")
    .setDropLast(false)
```

`OneHotEncoder` creates an additional column per each unique category in the original column.
Output columns "Geography" will be divided into 3 columns and "Gender" into 2 columns. 
That means we get 5 columns instead of these 2 original columns in the resulting table. 
Other feature columns so far stay unchanged.

### Merge to a Vector column

Before we normalize continues features, we need to merge individual columns into a single column.
That means the resulting column will look like:

```
row0: .. |DenseVector(42,2,0,1,1,1,101348.88)| ..
row1: .. |DenseVector(41,1,83807.86,1,0,1,112542.58)| ..
... etc.
```

```scala
// 3 - Merge to Vector
val continuesCols = List(
  "CreditScore",
  "Age",
  "Tenure",
  "Balance",
  "NumOfProducts",
  "EstimatedSalary"
)

val assembler = VectorAssembler()
  .setInputCols(continuesCols*)
  .setOutputCol("continues_features")
  .setInputSizes(List.fill(continuesCols.length)(1).map(Integer.valueOf)*)
```

By merging these columns into a single column, we slowly prepare training data for the final stage of the training pipeline, which is `LogisticRegression` model. 
Its input format requires to fit a single column with all features per row.

### Normalise Numbers and Merge Columns

```scala
// 4 - Normalise numbers
val standardScaler = StandardScaler()
  .setWithMean(true)
  .setInputCol("continues_features")
  .setOutputCol("continues_features_s")

// 5 - merge columns to features col
val categoricalCols = List("Geography", "Gender", "HasCrCard", "IsActiveMember")
val finalCols = categoricalCols :+ "continues_features_s"
// Geography is 3 countries, Gender is 2 + other 8 features
val encodedFeatures = List(3, 2)
val vectorSizes = encodedFeatures ++ 
  List.fill(categoricalCols.length - encodedFeatures.length)(1) :+ continuesCols.length
val finalAssembler = VectorAssembler()
  .setInputCols(finalCols*)
  .setOutputCol(featuresCol)
  .setInputSizes(vectorSizes.map(Integer.valueOf)*)
```

- `StandardScaler` transforms a column values using column mean and standard deviation values.
- `VectorAssembler` merges all input columns into one single column, which we name as `features`.

The resulting table format will have two columns:

```csv
features: DensVector | label: Double
```

## Train the Model

{{ resize_image(path="flink-ml/images/train-the-model.jpg",height=800, width=800,op="fit") }}
<p align="center">Figure 3. Model riding the train</p>

Now we combine all stages and train the model:

```scala
val lr = LogisticRegression()
  .setLearningRate(0.002d)
  .setLabelCol(exitedLabel)
  .setReg(0.1)
  .setElasticNet(0.5)
  .setMaxIter(100)
  .setTol(0.01d)
  .setGlobalBatchSize(64)

val stages = (List[Stage[?]](
    indexer,
    geographyEncoder,
    assembler,
    standardScaler,
    finalAssembler,
    lr
  )).asJava

val pipeline = Pipeline(stages)

val testSetSize = 2000
val totalSetSize = 10000
val trainSetSize = totalSetSize - testSetSize
val trainSet = trainData.limit(trainSetSize)
val testSet = trainData.limit(trainSetSize, testSetSize)

val pipelineModel = pipeline.fit(trainSet)
```

Our training pipeline consists of data preparation stages and `LogisticRegression` evaluator. 
`LogisticRegression` as the last stage is going to keep and train model weights.

Method `fit` executes the training loop. Input data goes through all the stages sequentially.
The `pipelineModel` variable is a trained model which we use further to assess its quality by calculating several metrics.

## Validate the Model

```scala
val validateResult = pipelineModel.transform(testSet)(0)

val resQuery =
  s"""|select 
      |$featuresCol, 
      |$exitedLabel as $labelCol, 
      |$predictionCol, 
      |rawPrediction        
      |from $validateResult""".stripMargin

val iter = tEnv.sqlQuery(resQuery).execute.collect
val firstRow = iter.next
val colNames = firstRow.getFieldNames(true).asScala.toList.mkString(", ")

val correctCnt = (List(firstRow).toIterable ++ iter.asScala).foldLeft(0) { 
  (acc, row) =>
    println(row)
    val label = row.getFieldAs[Double](labelCol)
    val prediction = row.getFieldAs[Double](predictionCol)
    if label == prediction then acc + 1 else acc
}
println(colNames)
println(
  s"correct labels count: $correctCnt, accuracy: ${correctCnt / testSetSize.toDouble}"
)
```

In the result, above calculation will print all the validate set rows with their predictions and final accuracy metric value:

```bash
.... < a lot of rows here>
+I[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.6078536710583327, -0.2776327360064674, 0.6985644677637127, -1.2297164068458775, -0.9064018047377373, -1.0106520912894439], 1.0, 0.0, [0.5157926901008845, 0.48420730989911553]]
features, label, prediction, rawPrediction
correct labels count: 1619, accuracy: 0.8095
```

Among 2000 validate set rows we got 1619 correctly labeled, which is 81% correctness of the trained model. 

One more useful evaluation of the model performance is to use Flink ML 
[BinaryClassificationEvaluator](https://nightlies.apache.org/flink/flink-ml-docs-release-2.3/docs/operators/evaluation/binaryclassificationevaluator/):

```scala
val evaluator = BinaryClassificationEvaluator()
    .setLabelCol(exitedLabel)
    .setMetricsNames(
      ClassifierMetric.AREA_UNDER_PR,
      ClassifierMetric.KS,
      ClassifierMetric.AREA_UNDER_ROC,
      ClassifierMetric.AREA_UNDER_LORENZ
    )

val outputTable = evaluator.transform(validateResult)(0)
val evaluationResult = outputTable.execute.collect.next
println(
  s"Area under the precision-recall curve: ${
    evaluationResult.getField(ClassifierMetric.AREA_UNDER_PR)}"
)
println(
  s"Area under the receiver operating characteristic curve: ${
    evaluationResult.getField(ClassifierMetric.AREA_UNDER_ROC)}"
)
println(
  s"Kolmogorov-Smirnov value: ${evaluationResult.getField(ClassifierMetric.KS)}"
)
println(
  s"Area under Lorenz curve: ${
    evaluationResult.getField(ClassifierMetric.AREA_UNDER_LORENZ)}"
)
```

We get the following values:
```bash
Area under the precision-recall curve: 0.3690456181301246
Area under the receiver operating characteristic curve: 0.6990527666047854
Kolmogorov-Smirnov value: 0.2932136963696369
Area under Lorenz curve: 0.6608346354166658
```

We are not going to try to improve current metrics and overall model performance. That would be a subject for another blog post.


## Save and Load Model

If we need to save learned model weights and then later load them again, Flink ML has special methods for that:

__Save Model data__:

```scala
pipelineModel.save("target/customer-churn-model/pipeline")
env.execute("Save PipelineModel")
```
Do not forget to call `execute` method to trigger model saving task to disk.

In the result we will get the following metadata and data on disk:

```bash
target/customer-churn-model/pipeline/
├── metadata
└── stages
    ├── 0
    │   ├── data
    │   │   └── part-a64dc5f8-aa9f-4926-b3c4-741046d6191b-0
    │   └── metadata
    ├── 1
    │   ├── data
    │   │   ├── part-820018d4-6bf6-4494-8de1-88e26b94054c-0
    │   │   └── part-8471e23f-af15-438e-9c10-a32e34ac9a64-0
    │   └── metadata
    ├── 2
    │   └── metadata
    ├── 3
    │   ├── data
    │   │   └── part-4008d3e0-259e-41a7-923f-12522d6a8950-0
    │   └── metadata
    ├── 4
    │   └── metadata
    └── 5
        ├── data
        │   └── part-aac4d931-8dea-4ee3-8610-884dc158e31d-0
        └── metadata
```        

__Load Model data__:

```scala
val model = PipelineModel.load(tEnv, "target/customer-churn-model/pipeline")
val validateResult = model.transform(validateSet)(0)
...
```

## Flink ML Summary

In result we were able to create Flink job which can learn and train ML model like `LogisticResgression`. As part of the learning process, we were able to 
prepare data in the proper format using Flink ML encoders and scalers. In case we want to use this model further, we can store and load the model state 
in the same or in a completely new Flink job. This allows us to train ML models in Flink on a specific environment and use them later in production.

