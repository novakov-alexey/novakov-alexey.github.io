+++
title="Scala FS2 - handle broken CSV lines"
date=2020-02-27
toc_enabled = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["fp", "fs2"]
categories = ["scala"]
+++
Recently, I ran into a familiar situation by doing data processing, where I needed to deal with a fragmented data stream. Having fragments, I had to detect manually where exactly new line/message starts and where current line/message ends in the stream. As turned out, one can aggregate intermediate state of the fragmented stream using scan function.

Let us dig down into how scan function is working.

{{ resize_image(path="scala-fs2/stream-fs2.jpeg", width=620, height=620, op="fit_width") }}
<!-- more -->

## Scan function

FS2 Stream Scan combinator works similar to scan function in Scala collection library. Official Scala doc for scan:

```scala
def scan[B >: A, That](z: B)(op: (B, B) ⇒ B)(
    implicit cbf: CanBuildFrom[List[A], B, That]
): That
```

says:
{% quote(author="Scala doc") %}
Computes a prefix scan of the elements of the collection.
{% end %}

Hm, what this would mean? It is much easier to try it and see how it works.

Here is [visual diagram](https://superruzafa.github.io/visual-scala-reference/scanLeft) for scan:
<img src="/img/blog/scanLeft.svg">

Let’s try with simple List using Scala REPL.

1. If we just aggregate and return acc every time in op function:

```scala
> val l = List("I", "am", "broken", "sentence")
l: List[String] = List("I", "am", "broken", "sentence"

> l.scan(""){case (acc, e) => acc + " " +  e}
res5: List[String] = 
List("", " I", " I am", " I am broken", " I am broken sentence")
```

2. If we apply some if condition based on the element content, then we can control, which element is going to be added into output collection:

```scala
> l.scan(""){
  case (acc, e) => if (e == "broken") e else  acc + " " +  e
}

res6: List[String] = 
List("", " I", " I am", "broken", "broken sentence")
```

As we can see, once boolean condition in the if is true, we return only the current element e and start new accumulation, i.e. produce next element without all the stuff we accumulated before. Hopefully, these two examples help to grasp an idea of the scan function.

## CSV file processing using FS2

Let’s first create a Scala IOApp using cats-effect library, which is coming as FS2 dependency, and define FS2 blocking execution context to work with files.

```scala
import java.nio.file.Paths
import java.util.concurrent.Executors
import cats.effect.{ExitCode, IO, IOApp, Resource}
import cats.instances.int._
import cats.syntax.all._
import fs2._

object Main extends IOApp {
  val Separator = ";"
  val isNextId: Regex = s"^(\\s*\\d+.*)$Separator".r
  val LastRowPaddingId = s"1$Separator"
  val ColumnsInFile = 10
  val HeaderLines = 1  
  
  override def run(args: List[String]): IO[ExitCode] =
    processFile("./input.csv").compile.drain.as(ExitCode.Success)  
    
  private val blockingExecutionContext =
    Resource.make(IO(
        ExecutionContext
          .fromExecutorService(Executors.newFixedThreadPool(2))
        ))(ec => IO(ec.shutdown()))  
        
  private def processFile(filePath: String): Stream[IO, Unit] = ???
}
```

Above **processFile** function will be returning a stream taking a file path to be processed as CSV file.

**processFile** implementation:

```scala
Stream.resource(blockingExecutionContext).flatMap { blockingEC =>
  io.file
    .readAll[IO](Paths.get(filePath), blockingEC, 4 * 4096)
    .through(text.utf8Decode)
    .append(Stream.eval(IO.pure("\n" + LastRowPaddingId)))
    .through(text.lines)
    .drop(HeaderLines)
    .scan(("", "")) {
      case ((acc, _), line) => concatBrokenLines(acc, line)
    }
    .filter { case (_, line) => line.trim.nonEmpty }
    .map { case (_, line) => line.split(Separator, ColumnsInFile) }
    .map(processRow)
    .foldMap(_ => 1)
    .map(n => println(s"Processed $n record(s)"))
}
```

The main code of finding the complete CSV line (a line, which can have multiple parts of one logical CSV line separated by line brakes like \n) starts at line:

```scala
.scan(("", "")) {
  case ((acc, _), line) => concatBrokenLines(acc, line)
}
.filter { case (_, line) => line.trim.nonEmpty }
```

Inside the scan function, we delegate ‘op’ part to concatBrokenLine function:

```scala
def concatBrokenLines(acc: String, line: String) = {  
  // next line detected, i.e. we flush `acc` downstream,
  // since it already contains a complete line to be processed
  if (isNextId.findFirstIn(acc).isDefined 
      && isNextId.findFirstIn(line).isDefined) (line, acc)
  
  // next line is not yet detected, i.e. we flush an empty string 
  // and append current line to the current `acc` state  
  else (acc + " " + line, "")
}
```

using regular expression **isNextId** we identify, whether new line marker is found. In the regular expression we want to find a number following by semicolon (according to current file business logic).

Read the inline comments in the _concatBrokenLines_ function on how using if/else logic, we control what needs to be put into next element of the downstream. As you can see, we use second half of the accumulator to push the complete line further (in the if branch).

**Now looking at scan and its concatBrokenLines function together**, we can summarise:

we process CSV lines by folding them via scan function using empty element as a tuple of two empty strings (“”, “”). In the head of the scanning lambda we use only first part of the zero element, we call it acc, i.e. accumulator. We also have **line** variable, which is given by the Stream.scan function itself. Then, we delegate the decision on what needs to be returned to the downstream using _filter_ function. Basically, we use _filter_ as a guard to control what actually needs to be passed further for the main processing logic as CVS line.

Also, we append fake line to be able to process the very last line. This last line needs one more marker to be properly detected as complete line:

```scala
.append(Stream.eval(IO.pure("\n" + LastRowPaddingId)))
```

## Test of the FS2 code

Using file input.csv:

```txt
index;City;population
1;Berlin is the
capital and largest city of
Germany by
both area and population.;3,748,148
1;Madrid is the capital of Spain and the largest
municipality in both the Community of Madrid and Spain as
a whole.;3,223,334
1;Donetsk former names: Aleksandrovka,
Hughesovka, Yuzovka, Stalino is an industrial city in Ukraine on the Kalmius
River.;929,063
```

we print the resulted lines of the stream in the _processRow_ function. In real life this function supposed to do something useful:

```scala
private def processRow(columns: Array[String]): Unit = 
  println(s"processed: ${columns.mkString(" :: ").trim}")
```

Output:

```txt
processed: 1 :: Berlin is the capital and largest city of Germany by both area and population. :: 3,748,148
processed: 1 :: Madrid is the capital of Spain and the largest municipality in both the Community of Madrid and Spain as a whole. :: 3,223,334
processed: 1 :: Donetsk former names: Aleksandrovka, Hughesovka, Yuzovka, Stalino is an industrial city in Ukraine on the Kalmius River. :: 929,063
Processed 3 record(s)
```

## Summary

Applying knowledge of functional combinators, we have gotten concise and clean code, without using any global mutable state outside of the stream definition. We have also solved the problem within the single data stream using scan to aggregate intermediate state and filter function as a guard to discard incomplete CSV lines.

FS2 library is very nice and especially having Cats and Cats-effect as direct dependency. 
See more examples for functional streaming in [FS2 guide](https://fs2.io/guide.html)

Source code: [https://github.com/novakov-alexey/fs2-csv-scan](https://github.com/novakov-alexey/fs2-csv-scan)