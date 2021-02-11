+++
title="Tensorflow Scala - Linear Regression via ANN"
date=2021-02-12
draft = true

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["deep learning", "machine learning", "tensorflow"]
categories = ["scala"]
+++

![TensorFlow Scala logo](https://raw.githubusercontent.com/eaplatanios/tensorflow_scala/master/docs/images/logo.svg?sanitize=true)

[TensorFlow Scala](https://github.com/eaplatanios/tensorflow_scala) is a strongly-typed Scala API for Tensorflow core C++ library developed by [Anthony Platanios](https://github.com/eaplatanios). This library
integrates with native Tensorflow library via [JNI](https://en.wikipedia.org/wiki/Java_Native_Interface), so no intermediate official/non-official Java libraries are used.
<!-- more -->
In this article we will implement "multiple linear
regression" from the [previos article](ann-in-scala-2/) about Customer Churn using Tensorflow.

# Setup Project

I am going to use SBT, but you can also use any other JVM-aware build tool.

New SBT project cofiguration:

```scala
lazy val tensorFlowScalaVer = "0.5.10" 

lazy val root = (project in file("."))
  .settings(
    name := "tensorflow-scala-example",
    libraryDependencies ++= Seq(
      "org.platanios" %% "tensorflow-data" % tensorFlowScalaVer,
      "org.platanios" %% "tensorflow" % tensorFlowScalaVer classifier "darwin"
    )
  )
```

For adding tensorflow-scala to existing project, just add two library dependencies from above configuration.

__Important:__ I am using OSX, so my classifier is `darwin`. In case you use Linux or Windows, please __change classifier__ value to currently available classifiers for those platforms, check it here to be sure: [https://repo1.maven.org/maven2/org/platanios/tensorflow_2.13/0.5.10/](https://repo1.maven.org/maven2/org/platanios/tensorflow_2.13/0.5.10/).


# Tensor API

Before start implementing an Artificial Neural Network with Tensorflow, let's
briefly look at how we can load data into a tensor. 

We would need to create matricies, so we can use scala collections and map
Arrays to Tensors and put them as rows into another Tensor. For example:

```scala
import org.platanios.tensorflow.api._

val t1 = Tensor(Tensor(1, 4, 0), Tensor(2, 3, 5))
println(t1.summarize())
```

will print:

```bash
Tensor[Int, [2, 3]]
[[1, 4, 0],
 [2, 3, 5]]
```

# Data Preparation

`TensorFlow Scala` also has data API to load datasets. We will use that API 
partly. Unfortunatelly, current documentation does not give an idea how to use Data API, so that we will use partly. Most of data preparation code I took from previos arcticle.

## Custom Code

```scala
type Matrix[T] = Array[Array[T]]

private def createEncoders[T: Numeric: ClassTag](data: Matrix[String])
  : Matrix[String] => Matrix[T] = {

  val encoder = LabelEncoder[String]().fit(TextLoader.column(data, 2))
  val hotEncoder = OneHotEncoder[String, T]().fit(TextLoader.column(data, 1))

  val label = t => encoder.transform(t, 2)
  val hot = t => hotEncoder.transform(t, 1)
  val typeTransform = (t: Matrix[String]) => transform[T](t)

  label andThen hot andThen typeTransform
}

def loadData() = {
  // loading data from CSV file into memory 
  val loader = TextLoader(Path.of("data/Churn_Modelling.csv")).load()
  val data = loader.cols[String](3, -1)

  val encoders = createEncoders[Float](data)
  val numericData = encoders(data)
  val scaler = StandardScaler[Float]().fit(numericData)

  // create data transformation as custom code
  val prepareData = (t: Matrix[String]) => {
    val numericData = encoders(t)
    scaler.transform(numericData)
  }

  // transform features
  val xMatrix = prepareData(data)    
  val yVector = loader.col[Float](-1)

```

## Tensforlow API

```scala
  // Wrap arrays into Tensors and set Shapes
  val xData = xMatrix.map(a => Tensor(a.toSeq)).toSeq
  val x = Tensor(xData).reshape(Shape(-1, features))
  val y = Tensor(yVector.toSeq).reshape(Shape(-1, targets))

  // use library API to split data for train and test sets
  val split = UniformSplit(x.shape(0), None)
  val (trainIndices, testIndices) = split(trainPortion = 0.8f)
  val xTrain = x.gather[Int](trainIndices, axis = 0)
  val yTrain = y.gather[Int](trainIndices, axis = 0)
  val xTest = x.gather[Int](testIndices, axis = 0)
  val yTest = y.gather[Int](testIndices, axis = 0)
  (xTrain, yTrain, xTest, yTest, prepareData)
}
```

As per comments above, we prepare training and test data and return it as 4
different Tensor objects. Also, we return a function as fith element of the tuple
for one more application (read further).

Full code of data preparation that includes:

- selecting specific CSV file columns
- normalazing numeric columns
- encoding categorical columns as one-hot 
- encoding label-like columns

see here:
 - [encoders](https://github.com/novakov-alexey/tensorflow-scala-example/blob/main/src/main/scala/encoders.scala)
 - [TextLoader](https://github.com/novakov-alexey/tensorflow-scala-example/blob/main/src/main/scala/TextLoader.scala) 


# Model Assembly

We are going to learn on 12 features to predict one target.

```scala
val features = 12
val targets = 1
val batchSize = 100

val input = tf.learn.Input(FLOAT32, Shape(-1, features))
val trainInput = tf.learn.Input(FLOAT32, Shape(-1, targets))
```

`batchSize` will be used in a couple of places of Tensorflow API.

In order to construct all 12 x 6 x 6 x 1 network below:

{{ resize_image(path="ann-in-scala-1/images/ann.png", width=600, height=600, op="fit") }}

we use the following API:

```scala
val layer =
  tf.learn.Linear[Float]("Layer_0/Linear", 6) >>
    tf.learn.ReLU[Float]("Layer_0/ReLU") >>
    tf.learn.Linear[Float]("Layer_1/Linear", 6) >>
    tf.learn.ReLU[Float]("Layer_1/ReLU") >>
    tf.learn.Linear[Float]("OutputLayer/Linear", 1) >>
    tf.learn.Sigmoid[Float]("OutputLayer/Sigmoid")
```

`layer` state is a composition of fully-connected layers with its own activation function. We specify `String` name for each layer that will be eventually used in Tensforflow graph.

```scala
val loss = tf.learn.L2Loss[Float, Float]("Loss/L2Loss")    
val optimizer = tf.train.Adam()
```

We are going to use `L2`, which is a `half least square error` as a loss function.
And Adaptive Moment Estimation (Adam) as weights optimization algorithm.

Finally, we pass above values to construct simple supervised model. 

```scala
val model = tf.learn.Model.simpleSupervised(
  input = input,
  trainInput = trainInput,
  layer = layer,
  loss = loss,
  optimizer = optimizer,
  clipGradients = ClipGradientsByGlobalNorm(5.0f)
)
```  

You can also create unsupervised model with `.unsupervised` method, if needed.