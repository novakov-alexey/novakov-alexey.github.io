+++
title="Convolutional Neural Network in Scala"
date=2021-04-02
draft = true

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["deep learning", "machine learning", "convolution", "computer vision", "image recognition"]
categories = ["scala"]
+++

Last time we used ANN to train a Deep Learning model for image recognition using MNIST dataset.
This time we are going to look to more advanced network called Convolutional Nueral Network or CNN in short.

CNN is designed to tackle image recognition problem. As we have seen last time, ANN using just hidden layers can learn quite well on MNIST.
However, for real life use cases we need higher accuracy. The main idea of CNN is to learn how to recognize object in their different shapes and positions 
using specific features of the image data. The goal of CNN is better model regularization by using filters.

CNN adds two more type of layers:

- Convolution layer
- Max/Average Pooling layer

Convolution is a mathematical [operation](https://en.wikipedia.org/wiki/Convolution), which is used in CNN layers instead of matrix multiplication like in fully-conected layers.
Typical CNN may consist of several Convolutional and Pooling layers. Final part of the network consists of the fully-connected layers like in ANN.

{{ resize_image(path="cnn-in-scala/images/cnn.png", width=800, height=800, op="fit_width") }}

# CNN Topology

We are going to design a simple CNN architecture for image recognition of the MNIST dataset. The problem type will be still the same:
predict a hand-written digit based on input image data. CNN works with image properties like height, width and color depth (RGB, etc.). 
Basically, it uses image pixel matrices to perform convolution and pooling operations. 
CNN works with every color channel separately. After some point, channels add up to each other using elementwise addition.

I am going to build CNN using my existing mini-library in Scala:

```scala
val cnn = Sequential[Precision, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByNorm(5.0),
    printStepTps = true
  )    
    .add(Conv2D(relu, 8, kernel = (5, 5), strides = (1, 1)))    
    .add(MaxPool(strides = (1, 1), pool = (2, 2), padding = false))       
    .add(Flatten2D())
    .add(Dense(relu, 50))      
    .add(Dense(softmax, 10))
```    

There are 3 more layers that we have not seen before. 

## Convolution layer

`Conv2D` is a Scala case class that has already familiar `relu` parameter as an activation function and other unique parameters for convolition:

 - `filterCount = 8` - number of filters which are going to be trained/optimized via gradient descent and back-propagation
 - `kernel  = (5, 5)` - window height and width to apply on input image/matrix
 - `strides = (1, 1)` - increment value when moving filter over the input image/matrix to the right `(1` and to the bottom `1)`

Implementing CNN we go into the world of 4-dimensional tensors. Added layer will keep its filters in 4D Tensor where:

1st dimension is filter count
2nd - color depth. Grey scale image is 1, RGB is 3, RGBA is 4 and so on.
3rd - filter height
4th - filter width

Added filter will have the following trainable weights and biases:
 - weights shape: (8 x 1 x 5 x 5) `Tensor4D`
 - biases shape: (8) `Tensor1D`

Convolution operation that above filter do will look like this:

<TODO PASTE IMAGE>

Every layer that we add to `Sequnetial` model has at least two methods `apply`, 
which is used to do forward propagation and `backward` for producing gradients based on the layer input data.

In Scala, we can code that like this:

```scala

// create regions with their position to be used by other functions
private def imageRegions(image: Tensor2D[T], kernel: (Int, Int), stride: (Int, Int)) =
  val (rows, cols) = image.shape2D    
  for i <- 0 to rows - kernel._1 by stride._1 yield   
  for j <- 0 to cols - kernel._2 by stride._2 yield       
    (image.slice((i, i + kernel._1), (j, j + kernel._2)).as2D, i, j)

// convolution operation which is
// elementwise multiplication between each image region and filter matrix
private def conv(filterChannel: Tensor2D[T], imageChannel: Tensor2D[T], 
         kernel: (Int, Int), stride: (Int, Int)) =    
  val filtered = 
    for row <- imageRegions(imageChannel, kernel, stride) yield
      for (region, _, _) <- row yield
        (region |*| filterChannel).sum

  filtered.as2D

// main forward function which is used by Conv2D layer to 
// apply every filter channel matrix to every input image.
// N.B. convoluted channels addds us together
private def forward(kernel: (Int, Int), stride: (Int, Int), 
  x: Tensor[T], w: Tensor[T], b: Tensor[T]): Tensor[T] =
  val (images, filters) = (x.as4D, w.as4D)    
   
  def filterImage(image: Array[Array[Array[T]]]) =
    filters.data.zip(b.as1D.data).map { (f, b) =>
      val filtered = f.zip(image).map { (fc, ic) =>
         conv(fc.as2D, ic.as2D, kernel, stride)
      }.reduce(_ + _)
      filtered + b.asT
    }
   
  images.data.par.map(filterImage).toArray.as4D

 // Layer interafce to the traning loop. 
 // `w` and `b` are the Layer state.
 override def apply(x: Tensor[T]): Activation[T] =
   val z = (w, b) match
     case (Some(w), Some(b)) => forward(kernel, strides, x, w, b)
     case _ => x // does nothing when one of the params is empty    
   val a = f(z)    
   Activation(x, z, a)
```

I left `private` modifier to show you that `apply` function uses other functions internally.

In general, forward propagation of convolution layer is not that simple as in Dense layer. 
Convolution layer is also computationaly intensive in forward and backward propagation.

The main line of code in forward propagatin is this one: `(region |*| filterChannel).sum`. It corresponds to elementwise multiplication of two matrices and summing the resulting matrix up
to get single number as one of the value for the output matrix. 

Tensor shape of the forward propagation can be calculated in advance using below formulas for height and width:
```scala
val rows = (height - kernel._1) / strides._1 + 1
val cols = (width - kernel._2) / strides._2 + 1
// final output shape after layer activation is
val shape = List(images, filterCount, rows, cols)
```

`images` is a number of images passed via `apply`, i.e. during the forward propagation. That means we can process a batch of images at once. 
We use similar idea in `Dense` layers via 2D Tensor to pass multiple images as rows at once (see further).

