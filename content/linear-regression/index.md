+++
title="Linear Regression using Neural Network"
date=2021-02-20
draft = true

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["deep learning", "machine learning", "linear regression"]
categories = ["scala"]
+++

In this article we are going to use [mini-library in Scala](/ann-in-scala-2) for Deep Learning
that we developed earlier in order to stidy basic linear regression task. 
We will learn model weigths using perceptron model, which will be our single unit network layer that emits target value. 
This model will predict a target value `yHat` based on two trained parameters: weight and bias. Both are scalar numbers.

```scala
y = bias + weight * x
```

# Data Preparation

Our goal is to show that perceptron model can learn the parameters, so that we can generate fake data:

```scala
import scala.util.Random

val random = new Random()
val weight = random.nextFloat()
val bias = random.nextFloat()

def batch(batchSize: Int): (ArrayBuffer[Double], ArrayBuffer[Double]) =
  val inputs = ArrayBuffer.empty[Double]
  val outputs = ArrayBuffer.empty[Double]
  (0 until batchSize).foldLeft(inputs, outputs) { case ((x, y), _) =>
    val input = random.nextDouble()
    x += input
    y += bias + weight * input 
    (x, y)
  }

val (xBatch, yBatch) = batch(10000)
val x = Tensor1D(xBatch.toArray)
val y = Tensor1D(yBatch.toArray)
val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)
```

We have prepared two datasets: 8000 data samples for train and 2000 samples for test cycles.

# Model Training

First, we initialize sequential model for just one dense layer with single unit which is going to be a perceptron model.

```scala
val ann = Sequential[Double, SimpleGD](
  meanSquareError,
  learningRate = 0.0002f,    
  batchSize = 16,
  gradientClipping = clipByValue(5.0d)
).add(Dense())
```

In order to avoid exploding gradient values, we also set grading clipping value, so that whenever our gradient is not
in `-5;5` numeric range it will be clipped to left or right boundary accordingly.

Let's start training and see if real weight and bias which we used to generate fake data are learnt by the 
percpetron:

```scala
val model = ann.train(xTrain.T, yTrain.T, epochs = 100)

println(s"current weight: ${model.weights}")
println(s"true weight: $weight")
println(s"true bias: $bias")
```

and - yes! Perceptron has learnt them!

```bash
running linearRegression 
epoch: 1/100, avg. loss: 0.5912029147148132
epoch: 2/100, avg. loss: 0.0827065110206604
epoch: 3/100, avg. loss: 0.0020780006889253855
epoch: 4/100, avg. loss: 7.106916018528864E-5
epoch: 5/100, avg. loss: 4.363957486930303E-5
epoch: 6/100, avg. loss: 3.614391971495934E-5
epoch: 7/100, avg. loss: 2.956729986181017E-5
epoch: 8/100, avg. loss: 2.413452421023976E-5
epoch: 9/100, avg. loss: 1.9695862647495233E-5
epoch: 10/100, avg. loss: 1.607407466508448E-5
epoch: 11/100, avg. loss: 1.311903542955406E-5
epoch: 12/100, avg. loss: 1.0707839464885183E-5
epoch: 13/100, avg. loss: 8.740245903027244E-6
epoch: 14/100, avg. loss: 7.134528914320981E-6
epoch: 15/100, avg. loss: 5.824046638736036E-6
epoch: 16/100, avg. loss: 4.754453129862668E-6
epoch: 17/100, avg. loss: 3.88142188967322E-6
epoch: 18/100, avg. loss: 3.168797093167086E-6
epoch: 19/100, avg. loss: 2.587080643934314E-6
epoch: 20/100, avg. loss: 2.112206175297615E-6
epoch: 21/100, avg. loss: 1.7245364460904966E-6
epoch: 22/100, avg. loss: 1.4080474102229346E-6
epoch: 23/100, avg. loss: 1.1496620118123246E-6
epoch: 24/100, avg. loss: 9.387074442201992E-7
epoch: 25/100, avg. loss: 7.664730219403282E-7
epoch: 26/100, avg. loss: 6.258487132981827E-7
epoch: 27/100, avg. loss: 5.11030918914912E-7
epoch: 28/100, avg. loss: 4.172821661541093E-7
epoch: 29/100, avg. loss: 3.4073505617016053E-7
epoch: 30/100, avg. loss: 2.7823239179269876E-7
...
epoch: 90/100, avg. loss: 1.4646350538632946E-12
epoch: 91/100, avg. loss: 1.1960714536851658E-12
epoch: 92/100, avg. loss: 9.767531835430665E-13
epoch: 93/100, avg. loss: 7.976503572232341E-13
epoch: 94/100, avg. loss: 6.513887736495083E-13
epoch: 95/100, avg. loss: 5.319465571154702E-13
epoch: 96/100, avg. loss: 4.3440592919705145E-13
epoch: 97/100, avg. loss: 3.5475089662714843E-13
epoch: 98/100, avg. loss: 2.8970182914915543E-13
epoch: 99/100, avg. loss: 2.365805305565988E-13
epoch: 100/100, avg. loss: 1.931998262543963E-13
current weight: List(
(
weight = sizes: 1x1, Tensor2D[Double]:
[[0.6741596449637218]]
,
bias = sizes: 1, Tensor1D[Double]:
[0.6995659301084302]
,
f = no-activation,
units = 1))
```

I have cut some output in the middle, but it does not hard to see the progress.

Latest model weights:

```bash
[[0.6741596449637218]]
[0.6995659301084302]
```

Original/true weights we used for data generation:

```bash
true weight: 0.674161
true bias: 0.6995651
```

# Test dataset

```scala
val testPredicted = model.predict(xTest)  
val value = meanSquareError[Double].apply(yTest.T, testPredicted)
println(s"test meanSquareError = $value")
```

Loss value on test is quite close to training loss:

```bash
test meanSquareError = 1.7679944726888933E-13
```

# Visualization

Let's visualize our loss function using nice library [Picta](https://acse-fk4517.github.io/picta-docs/index.html).
We are going to use Picta in [Jupyter notebook](https://jupyter.org/) via [Almond Scala kernel](https://almond.sh/).

Before we try to use use 2D or 3D [Canvas API](https://acse-fk4517.github.io/picta-docs/canvas.html)
 from `Picta`, we need to prepare metrics data. 
We could run the entire training code in Jupyter together with Picta around, but as of now Almond Scala kernel
does not support Scala 3 which was used to write the Deep Learning code. So we will go with CSV files to 
bridge two worlds.

This is our plan:

1. Store metrics data from existing Scala 3 code to CSV files
2. Use CVS files in Jupyter with Scala 2.13

## Data: Loss metric per epoch

```scala
val lossData = model.losses.zipWithIndex.map((l,i) => List(i.toString, l.toString))
store("metrics/lr.csv", "epoch,loss", lossData)
```

```csv
epoch,loss
0,0.5912029147148132
1,0.0827065110206604
2,0.0020780006889253855
3,7.106916018528864E-5
4,4.363957486930303E-5
...
```

`store` function is just cretaing CSV file out of data in the Scala list:

```scala
import scala.util.Using
import java.io.File
import java.io.PrintWriter

def store(filename: String, header: String, data: List[List[String]]) =    
  Using.resource(new PrintWriter(new File(filename))) { w =>
    w.write(header)
    data.foreach { row =>      
      w.write(s"\n${row.mkString(",")}")        
    }
  }
```

This is the first CSV file that contains loss value pe training epoch.

## Data: Loss Function Surface

`Picta` can also draw 3D plots, so that we can generate loss surface based on weight and bias parameters (`x` and `y` axis) and loss value as `z` axis.

```scala
val weights = for (i <- -100 until 100) yield i + 1.5 * random.nextDouble
val biases = weights // we use the same range for bias
  
val losses = weights.par.map { w =>
  val wT = w.as0D
  biases.foldLeft(ArrayBuffer.empty[Double]) { (acc, b) =>
    val loss = ann.loss(x.T, y.T, List(Weight(wT, b.as0D)))  
    acc :+ loss
  }
}
 
val metricsData = weights.zip(biases).zip(losses)
  .map { case ((w, b), l) =>
    List(w.toString, b.toString, l.mkString("\"", ",", "\"")) 
  }
  
store("metrics/lr-surface.csv", "w,b,l", metricsData.toList)
```

```csv
w,b,l
-98.87702045894542,-98.87702045894542,"23171.59049703857, // here come 201 values for column `l` which stands for loss.
```

Last column is going to be used in 3D plot as `Z` axis. It is a list rather than a scalar value. This way we can draw
a surface in Picta later.

## Loss vs. Epoch Plot

1. Add library, change notebook output.

```scala
import $ivy. `org.carbonateresearch::picta:0.1.1`
// required to initialize jupyter notebook mode
import org.carbonateresearch.picta.render.Html.initNotebook
initNotebook() // stops standard output
```

2. Load data from CVS, transform strings to doubles.

```scala
import org.carbonateresearch.picta.IO._
import org.carbonateresearch.picta._

val metricsDir = getWorkingDirectory + "/../metrics"
val filepath = metricsDir + "/lr.csv"
val data = readCSV(filepath)
val epochs = data("epoch").map(_.toInt)
val losses = data("loss").map(_.toDouble)
```

3. Create data series, chart and plot data

```scala
val series = XY(epochs, losses).asType(SCATTER).drawStyle(LINES)
val chart = Chart()
    .addSeries(series.setName("Learning loss"))
    .setTitle("Linear Regression Example: Loss vs. Epoch")
chart.plotInline
```

{{ resize_image(path="linear-regression/loss-versus-epoch.png", width=800, height=600, op="fit") }}

## Loss Function Surface

Similar code is to build 3D chart. Now we hava `z` axis which is feed as flat list, but with surface of 201 points, which we set in `n` parameter.

```scala
val data = readCSV(s"$metricsDir/lr-surface.csv")
val w = data("w").map(_.toDouble)
val b = data("b").map(_.toDouble)
val loss = data("l").map(_.split(",").map(_.toDouble))

val surface = XYZ(x=w, y=b, z=loss.flatten, n=loss(0).length).asType(SURFACE)
val surfaceChart = Chart()
  .addSeries(surface)
  .setTitle("Loss Function Surface")
  .addAxes(
    Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss")
  )
surfaceChart.plotInline
```

I have created several print-screen just to show you this beatiful surface from different angels:

{{ resize_image(path="linear-regression/loss-surface.png", width=800, height=600, op="fit") }}
{{ resize_image(path="linear-regression/loss-surface-2.png", width=800, height=600, op="fit") }}
{{ resize_image(path="linear-regression/loss-surface-3.png", width=800, height=600, op="fit") }}
{{ resize_image(path="linear-regression/loss-surface-4.png", width=800, height=600, op="fit") }}

### Contour Chart

One more fancy chart from Picta is Contour chart, which sometimes can be useful for analysis.

```scala
val contour = XYZ(x=w, y=b, z=loss.flatten, n=loss(0).length).asType(CONTOUR)
val contourChart = Chart().addSeries(contour).setTitle("Loss Contour")
             .addAxes(Axis(X, title = "w"), Axis(Y, title = "b"), Axis(Z, title = "loss"))

contourChart.plotInline
```

{{ resize_image(path="linear-regression/loss-contour.png", width=800, height=600, op="fit") }}


Just for you to proove that this drawn in Jupyter actually :-)

{{ resize_image(path="linear-regression/jupyter-view.png", width=800, height=600, op="fit") }}

Again big thanks to [Almond project](https://github.com/almond-sh/almond) that made Scala easily available in Jupyter.

# Summary

We have seen that our perceptron model is able to learn weights very quick for simple 1 input variable.
So it proves that gradient decsent algorithm implemeted earlir is working fine.

Also, we could visualize loss metrics using Picta and Almond Jupyer kernel for Scala quite easily
that can help us to tune model training in real life use cases.