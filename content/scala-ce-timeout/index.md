+++
title="Cats-Effect: Cancel Scala Process on Timeout"
date=2020-02-29
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["fp", "cats-effect"]
categories = ["scala"]
+++

{{ resize_image(path="scala-ce-timeout/maico-amorim.jpg", width=600, height=600, op="fit") }}

Sometimes Scala developer needs to call external program, which is running outside of the JVM. 
In this case, we use `scala.sys.process` package. Process package has bunch of functions to spin up new processes, 
consume their outputs and errors. Also, spawned process can be stopped. Usually, we run external programs for a short period
of time to make some side-effect. Then, we analyse its exit code to apply some error handling logic in our main Scala program.
It worth to say that process API is blocking execution thread, when we are waiting for its completion. To summarise, Scala
developer wants to do the following:
<!-- more -->
1. Start external program as a process by giving a string containing command to be executed in underlying operating system.
2. Wait for completion and get the exit code.
3. Cancel spawned process, in case waiting time for its completion is greater than a certain threshold.

Good news, we can do all of that in Cats-Effect leveraging IO monad to handle a side-effect and having timeout logic around.


## Handle blocking code

Besides usual `ContextShift`, we will use separate thread pool to run blocking code of process API.
Cats-Effect provides `Blocker` class to evaluate specific IO on a given execution context. In case below, we are going
to use CachedThreadPool, which can grow almost infinitely. Our main execution context will be still `ExecutionContext.global`
and it will be used for non-blocking operations.

```scala

import java.util.concurrent.Executors
import java.util.concurrent.TimeoutException

import cats.effect.ExitCase._
import cats.effect.{Blocker, Concurrent, ContextShift, IO, Timer}
import cats.implicits._

import scala.concurrent.ExecutionContext
import scala.concurrent.duration._
import scala.sys.process._

implicit val timer: Timer[IO] = IO.timer(ExecutionContext.global)
implicit val cs: ContextShift[IO] = IO.contextShift(ExecutionContext.global)

val cachedThreadPool = Executors.newCachedThreadPool()
val blocker = Blocker.liftExecutionContext(
    ExecutionContext.fromExecutor(cachedThreadPool)
)

```

## Blocking Task

`Blocker` has `blockOn` method, which takes an IO and returns an IO to be evaluated on the specified earlier thread pool.

```scala
def startProcess(cmd: String): IO[Int] = {
  val blockingTask = blocker.blockOn(IO(cmd.run()))

  //.... tbd
}

```

Using Cats-Effect [Bracket](https://typelevel.org/cats-effect/typeclasses/bracket.html) type class we can safely start our
process and handle its IO cancelation. On task cancel event, we are going to call `Process#destroy` method to stop running in OS.

```scala
def startProcess(cmd: String): IO[Int] = {
  val blockingTask = blocker.blockOn(IO(cmd.run()))
  blockingTask.bracketCase { p =>
    IO(p.exitValue())
  } { (p, exit) =>
    exit match {
      case Completed => IO.unit
      case Error(_) | Canceled => IO(p.destroy())
    }
  }
} 
```

Above pattern matching case on Canceled, we stop process `p` using `destroy()`.

## Run a task with timeout

One of the way to run Cats IO with timeout is to use its `race` method from `Concurrent` type class. Second
task in race is a call of `Timer#sleep`, which is semantically blocking an IO for a specified duration.

Let's bring special function to start a race for two IOs and have third task as fallback IO, in case first IO
was not completed before timeout. Below function was reused from Cats-Effect documentation:

```scala
def timeoutTo[F[_], A](
    fa: F[A],
    after: FiniteDuration,
    fallback: F[A]
  )(implicit timer: Timer[F], concurrent: Concurrent[F]): F[A] = {

    concurrent.race(fa, timer.sleep(after)).flatMap {
      case Left(a) =>
        println("Done")
        concurrent.pure(a)
      case Right(_) =>
        println("Timeout")
        fallback
    }
  }
```

Now we are ready to run our blocking task with timeout. For the sake of example, we set 1 second as timeout and failing returned IO,
by giving fallback IO with exception. Let us run infinitely running command such `tail -f` on some file to simulate 
long-running task, which we need to cancel in case of timeout.

```scala
val task = startProcess("tail -f build.sbt")
val finalTask = timeoutTo(task, 1.second, 
  IO.raiseError(new TimeoutException("Failed to run external process")))
finalTask.unsafeRunSync()
```

Output
```scala
scala> finalTask.unsafeRunSync()
          //... here comes a content of build.sbt as per given command 
  
Timeout
java.util.concurrent.TimeoutException: Failed to run external process
```

Happy case:

```scala
scala> val task = startProcess("echo cats")
scala> val finalTask = timeoutTo(task, 1.second, 
  IO.raiseError(new TimeoutException("Failed to run externall process")))
scala> finalTask.unsafeRunSync()
cats
Done
res3: Int = 0
```

## Summary

- Using `Bracket` we can easily catch IO cancelation and release acquired resource. 
In the example above, we destroy external process, so that OS resource is released. 
- Cats `Blocker` helps us to run blocking tasks safely with regards
to other non-blocking tasks. 
- And `IO.race` can be used to simulate timeout, since it cancels race looser.

## Misc
<ul id="frontmatter" class="frontmatter frontmatter_page"><li>Photo by Maico Amorim on Unsplash</li></ul>