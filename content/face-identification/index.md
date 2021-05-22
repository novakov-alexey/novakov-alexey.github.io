+++
title="Face Identification with VGGFace and OpenCV"
date=2021-05-22
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["deep learning", "machine learning", "computer vision", "cnn"]
categories = ["scala"]
+++

Face detection and recognition is one the area where Deep Learning is incredibly useful. There are many studies and datasets related to
human faces and their detection/recognition. In this article we will implement Machine Learning pipeline for face detection and recognition using few libraries and CNN model.

# Pipeline

One the part will be implemented with very popular C++ library [OpenCV](https://opencv.org/), which is around for a long time. 
It has many modules for image processing, object classification, neural networks and more. We are going to use its Java Wrapper - [JavaCV](https://github.com/bytedeco/javacv)

OpenCV comes with human face detector module, which is called "Haar Cascade" classifier. 
This class takes an image and returns a [Rect](http://bytedeco.org/javacpp-presets/opencv/apidocs/org/bytedeco/opencv/opencv_core/Rect.html) object of the detected face(s) in it. The Rect object is a data structure that has X and Y coordiantes of
the left-top corners plus width and height where region of interest is located. In our case, rectangular area is a face. For example:

Original photo:
{{ resize_image(path="face-identification/tom_photo.jpg", width=800, height=600, op="fit") }}

Cropped with Haar Cascade:
{{ resize_image(path="face-identification/tom_face.jpg", width=800, height=600, op="fit") }}


Once we get an area of the detected face(s), we can compare its pixel data with in advance extracted face features of known faces. 
Comparison algroithm calculates a Euclidean distance between detected face vector with vectors of known faces. 
The smallest distance with some known face sets its label as a result. 
That means, we can take known face label that has smallest distance as a result of face identification. 
Our workflow will look like this:

{{ resize_image(path="face-identification/workflow.png", width=800, height=600, op="fit") }}

# Photo Preparation

In order to extract person features, we will prepare separate folder with image per each person:

- Alexey (myself)
- [Guy Ritchie](https://en.wikipedia.org/wiki/Guy_Ritchie)
- [Tom Araya](https://en.wikipedia.org/wiki/Tom_Araya)

Input directory:

```bash
raw_photos
├── guy_ritchie
└── tom_arraya
```

Output folder:

```bash
dataset-people
├── guy_ritchie
└── tom_arraya
```

{{ resize_image(path="face-identification/guy_ritchie.png", width=800, height=600, op="fit") }}
{{ resize_image(path="face-identification/tom_araya.png", width=800, height=600, op="fit") }}

Plus I am adding my photos to `dataset-people` directory

```bash
dataset-people
├── alexey
├── guy_ritchie
└── tom_arraya
```

Below program reads photos from the local folder and crops faces to save them to a separate folder:

```scala
//cropFaces.scala

import org.bytedeco.opencv.opencv_core.{Rect, Mat, Size}
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.global.opencv_imgcodecs.{imread, imwrite}

import java.nio.file.{Files, Paths}

def createIfNotExists(path: String) =
  if !Files.exists(Paths.get(path)) then 
    Files.createDirectory(Paths.get(path))

@main
def crop() =
  val datasetDir = "dataset-people"
  createIfNotExists(datasetDir)

  val dirs = Paths.get("raw_photos").toFile.listFiles.filter(f => !f.getName.startsWith("."))

  for dir <- dirs do
    val label = dir.getName
    println(s"Extracting faces for '$label' label")

    createIfNotExists(Paths.get(datasetDir, label).toString)
    val images = dir.listFiles.filter(_.toString.endsWith(".jpg"))

    for file <- images do
      println(s"Reading file: $file")
      val image = imread(file.toString)
      val faces = detectFaces(image)  

      for (face, i) <- faces.get.zipWithIndex do        
        val crop = Rect(face.x, face.y, face.width, face.height)
        val cropped = Mat(image, crop)
        resize(cropped, cropped, Size(ImageHeight, ImageWidth))  
        val filename = Paths.get(datasetDir, label, s"$i-${file.getName}").toString
        println(s"Writing $filename")
        imwrite(filename, cropped)
```

The same code on GitHub is [here](https://github.com/novakov-alexey/face-identification/blob/main/src/main/scala/cropFaces.scala).

# Get ONNX model

Now we can extract face features for all three persons. We are going to use CNN model which was trained on the [VGGFace dataset](https://exposing.ai/vgg_face/).
The easiest option to access Keras (which is Python high-level API to Tensorflow) trained model from Scala is to export it to [ONNX](https://onnx.ai/) format. 
Let's proceed:

1. Create `SavedModel` file:
- Instaniate Python VGGFace class from [keras-vggface](https://github.com/rcmalli/keras-vggface) library
  with `include_top=False` to skip last layer of the CNN. 
- Save instaniated model to Tensforflow [Tensorflow SavedModel](https://keras.io/api/models/model_saving_apis/)

```python
from keras_vggface.vggface import VGGFace

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
COLOR_CHANNELS = 3

output_path = 'vggface_model'
model = VGGFace(model='vgg16',
                include_top=False,
                input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                pooling='avg')
model.save(output_path)
```


**Important: we save VGGFace model without last layer. We do not need any predictions that VGGFace model originally was supposed to output, 
since we are going to use extracted features to predict other persons than those in original VGGFace dataset.**

2. Convert saved model to ONNX format via [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) library
```bash
python -m tf2onnx.convert \
  --saved-model vggface_model \
  --output data/model.onnx \
  --tag serve
```

After converting SavedModel to ONNX we get file `data/model.onnx` of approx. 56Mb in size:

 # Extracting Features

 Now we can use ONNX model from Scala code. In this step, we use VGGFace model to extract features of all 3 person faces
and save them into a file as HasMap for further step.

First we implement common functions, which we will use one more time for real-time face identification algorithm:

 ```scala
 // common.scala
import io.kjaer.compiletime.*

import org.bytedeco.javacpp.indexer.{UByteIndexer, FloatRawIndexer}
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.opencv_core.{Mat, Scalar, RectVector, UMat}
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier

import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*

import java.nio.file.{Files, Paths, Path}
import io.bullet.borer.Cbor
import java.io.{ByteArrayOutputStream, File}

val OutputSize: Dimension = 512
val FeatureFilePath = "data/precomputed_features.cbor"

type Features = Map[String, Array[Float]]

def saveFeatures(features: Features) =
  val file = File(FeatureFilePath)
  Cbor.encode(features).to(file).result

def getModel(path: Path = Paths.get("data", "model.onnx")) =
  val bytes = Files.readAllBytes(path)
  ORTModelBackend(bytes)

def predict(
  images: Array[Float], 
  model: ORTModelBackend, 
  batch: Dimension = 1, 
  outputSize: Dimension = OutputSize
) =
  val input = Tensor(images, tensorDenotation, tensorShapeDenotation, shape(batch))
  model.fullModel[
    Float, 
    "ImageClassification", 
    "Batch" ##: "Features" ##: TSNil, 
    batch.type #: outputSize.type #: SNil](
      Tuple(input)
    )

def scale(img: Mat): Mat =
  val out = Mat()
  img.assignTo(out, CV_32FC4)
  subtract(out, Scalar(93.5940f, 104.7624f, 129.1863f, 0f)).asMat

def toArray(mat: Mat): Array[Float] =
  val w = mat.cols
  val h = mat.rows
  val c = mat.channels
  val rowStride = w * c

  val result = new Array[Float](rowStride * h)
  val indexer = mat.createIndexer[FloatRawIndexer]()
  var off = 0
  var y = 0
  while (y < h)
    indexer.get(y, result, off, rowStride)
    off += rowStride
    y += 1

  result
```

At this point we can run extraction step:

```scala
//extract.scala
import io.kjaer.compiletime.*
import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.opencv_core.{Mat, Scalar}

import java.nio.file.{Files, Paths}
import java.io.File
import javax.imageio.ImageIO
import scala.collection.parallel.CollectionConverters.*

def label2Features(dirs: Array[File]) = 
  lazy val model = getModel()
  val batchSize = 16

  dirs.par.map { dir =>
    val label = dir.getName
    println(s"Extracting features for '$label' at $dir folder")

    val groups = dir.listFiles.grouped(batchSize)
    val features = groups.map { files =>
      val images = files.map(f => toArray(scale(imread(f.toString)))).flatten
      val currentBatch = files.length.asInstanceOf[Dimension]      
      val out = predict(images, model, currentBatch)      
      out.data.grouped(OutputSize).toList
    }  
    label -> features.toList.flatten
  }

@main
def extract =
  val dirs = Paths.get("dataset-people").toFile.listFiles

  val avgFeatures = label2Features(dirs).map {
    (label, features) => 
      val count = features.length      
      val sum = features.reduce((a, b) => a.zip(b).map(_ + _))
      label -> sum.map(_ / count)
  }.toList.toMap

  saveFeatures(avgFeatures)
 ``` 

 In the `extract` function, we do element-wise addition of all extracted vectors per each person. 
 Then, final person vector is divided by number of images for that person to get average values of the extracted features. 
 In the result we will get/define such type alias:

 ```scala
 type Features = Map[String, Array[Float]]
 ```

 # Identify Faces

 Finally, we can use our extracted features to identify a person face. Keep in mind, that our prediction algorithm
 knowns only 3 person faces. Any other person may be confused with one of the three known faces or they can be unknown.
 If we want to more different people to be identifieable, we need their features as well, so go to step number one of our 
 pipeline to collect those people photos and extract their features.

 ```scala
 // main.scala
import org.bytedeco.opencv.opencv_core.{Mat, Size, Rect, Point, Scalar}
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_videoio.VideoCapture
import org.bytedeco.javacv.{CanvasFrame, OpenCVFrameConverter}

import javax.swing.WindowConstants

def createCavasFrame = 
  val frame = CanvasFrame("Detected Faces")
  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  frame.setCanvasSize(1280, 720)
  frame

def calcLabel(face: Array[Float], features: Features, threshold: Int = 100) = 
  features.foldLeft("?", Float.MaxValue){ 
    case ((label, min), (l, f)) => 
      val d = distance(face, f)
      if d < threshold && d < min then (l, d)
      else (label, min)
  }._1  

def drawLabel(label: String, frame: Mat, topLeft: Point) =
  val x = math.max(topLeft.x - 10, 0)
  val y = math.max(topLeft.y - 10, 0)
  val font = FONT_HERSHEY_SIMPLEX
  val thickness = 2
  val fontScale = 1.0
  val baseline = new Array[Int](2)
  val size = getTextSize(label, font, fontScale, thickness, baseline)
  val rectColor = Scalar(255, 0, 0, 0)
  rectangle(
    frame,
    Point(x, y - size.height() - thickness),
    Point(x + size.width() - thickness, y + 10),
    rectColor,
    CV_FILLED,
    LINE_8,
    0)
  val fontColor = Scalar(0, 255, 0, 0)
  putText(frame, label, Point(x, y), font, fontScale, fontColor, thickness, CV_FILLED, false)

def toModelInput(crop: Rect, frame: Mat) =
  val cropped = Mat(frame, crop)
  resize(cropped, cropped, Size(ImageHeight, ImageWidth))
  toArray(scale(cropped))

def drawRectangle(face: Rect, frame: Mat) =
  rectangle(frame,
    Point(face.x, face.y),
    Point(face.x + face.width, face.y + face.height),
    Scalar(0, 255, 0, 1)
  )
 ```

 In the main function we run infinite loop that captures video frame as an image.
 The captured image is then used to detect and identify person faces.

 ```scala
 @main
def demo() =
  val capture = VideoCapture(0)
  val canvasFrame = createCavasFrame  
  val frame = Mat()
  val converter = OpenCVFrameConverter.ToMat()
  val model = getModel()
  val features = loadFeatures

  try
    while capture.read(frame) do
      val faces = detectFaces(frame)
      
      for face <- faces.get yield
        drawRectangle(face, frame)
        val crop = Rect(face.x, face.y, face.width, face.height)
        val image = toModelInput(crop, frame)
        val faceFeatures = predict(image, model).data
        val label = calcLabel(faceFeatures, features)
        drawLabel(label, frame, crop.tl)

      canvasFrame.showImage(converter.convert(frame))                              
  finally
    capture.release
    canvasFrame.dispose
 ```

## Demo time


{{ resize_image(path="face-identification/guy_identity.png", width=800, height=600, op="fit") }}
{{ resize_image(path="face-identification/tom_identity.png", width=800, height=600, op="fit") }}

If I put more faces into the frame, identification algorithm may confuse some of them and identify someone with a beard as Tom Araya or someone
with bright white skin color as Guy Ritchie. In order to overcome such issue, you need to add more different faces. 
Also, we would need to tune `threshold` parameter, which is used to discard faces which are far away from those we are interested in.
Level of certantity is relative of course, there can be still many people in the world who have very similar face features like Tom, Guy or me.

# Summary

We have made powerful application with so little code to identify some person faces. There are several libraries were used to 
get the face detection working, such as OpenCV. [ONNX-Scala](https://github.com/EmergentOrder/onnx-scala) to access ONNX model fromm Scala. 
[Borer](https://sirthias.github.io/borer/) library to save and load face features as Scala object (HashMap) into memory from disk.

Current approach to identify faces by calculating Euclidian distance between input faces and pre-calculated face is not the only one.
We could also train custom CNN or VGGFace-based model with new layer to predict labels for Tom, Guy and myself. 
However, such approach is compute intensive and actually gave me quite bad results. 
If you know something crucial about this approach to work well, please let me know.

Other approaches to solve face identification task which you may want to get familiar with are
[Triplet Loss function](https://machinelearning.wtf/terms/triplet-loss/) and [Siamese Networks](https://machinelearning.wtf/terms/siamese-neural-network/). 
Perhaps, I will try one of them next time.

# Links

You can find complete project code at GitHub:

[https://github.com/novakov-alexey/face-identification/tree/main/src/main/scala](https://github.com/novakov-alexey/face-identification/tree/main/src/main/scala)
