+++
title="Monads in Scala"
date=2020-03-28
toc_enabled = false
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["fp"]
categories = ["scala"]
+++
Once you start dig deeper into Scala and its suitability for functional programming, you meet Monads. In this blog post, we will explore Monads in Scala:
their usage and usefulness. 

{{ resize_image(path="scala-monads/flatmap-all-the-things.png", width=620, height=620, op="fit_width") }}
<!-- more -->
## What is Monad?

You have probably already heard this quote:

{% quote(author="Saunders Mac Lane") %}A monad is just a monoid in the category of endofunctors{% end %}

[More details on StackOverflow answer](https://stackoverflow.com/a/3870310/6176274)

Well, that does not bring much help. Obviously, Monad is not just Scala pattern, but it is something what is coming 
from [Category Theory](https://en.wikipedia.org/wiki/Category_theory). 
However, we are not going to touch Category Theory in general, but let's say that Monad definition is coming from abstract theory of Mathematics.

I like another definition of Monad, which is given in the  book "Functional Programming in Scala":

{% quote(author="Chiusano, Bjarnason") %}Monad is an abstract interface{% end %}

It is more clear for programmers. Before we clarify in details what Monad is, let us look at some examples of Mondas in Scala standard library.
This might already click for you that Monad is not something from aliens:

- Option
- Either
- List
- Future
- Map
- Set
- Stream
- Vector
- Try 

... and others

## What makes thing a Monad?

There are several minimum combinations of functions which make some type a Monad. One of the popular minimum set is two functions:

- **flatMap** - also known as `bind`
- **unit** - also known as `pure` in [Cats library](https://typelevel.org/cats/typeclasses/monad.html#monad-instances) or `apply` in pure Scala

These two functions implemented for some type bring powerful abstraction to write complex programs easy.

## Make your Monad

Monad sometimes reminds a container to work with its values using special interface. If we model Monad ourselves, then it may look like a box with a thing
inside, which we access using `flatMap` and one more useful function `map`:

```scala
class Box[A](v: A) { 

  def flatMap[B](f: A => Box[B]): Box[B] = f(v)
  
  def map[B](f: A => B): Box[B] = flatMap(a => new Box(f(a)))
}
```

**map** - is implemented in terms of `flatMap` + `unit` (i.e. Box class constructor). 
So we can implement `map` for any kind of Monads, as we will see that later.

Now let's use `Box ` Monad to show some usage example:

```scala
scala> new Box(1)
res2: Box[Int] = 1

scala> res2.map(_ + 1)
res3: Box[Int] = 2

scala> res3.flatMap(i => new Box(1 + i))
res5: Box[Int] = 3
```

`Box` contains single integer value and allows us to manipulate it without leaving `Box` context, i.e. our result is always a `Box[T]`.
We can also make variable `v` as public and read it when needed. `Box` behaves similarly to non-empty single element list.
It is hard to say when this particular `Box` Monad will be useful looking at above example. However, it should give you an idea how Monad implementation may look like.

## Scala Examples

**List**

List operates on collection of values. 

```scala
scala> val l = List(1,2,3) // <-- unit
l: List[Int] = List(1, 2, 3)

scala> l.map(_ + 1)
res0: List[Int] = List(2, 3, 4)

scala> l.flatMap(i => List(i + 1))
res1: List[Int] = List(2, 3, 4)
```

**Option**

Option has two sub-types: `Some` and `None`. 

`Some` is like non-empty single element list, similar to Box Monad example above.
`None` ignores application of lambda function in `flatMap` or `map`.

```scala
val isOn = Some(1) // <-- unit
val isBlack = None // <-- unit without any argument

def makeCoffee: Option[String] = Some(1)

scala> isOn
         .flatMap(_ => isBlack
         .flatMap(_ => makeCoffee))

res0: Option[String] = None
```

Example above won't return value of `isOn` variable because the first `flatMap` call returns `None` because of `isBlack`, so that second `flatMap` even won't be called.

## Generic Monad

We have already seen example of at least 3 Monads above. In order to detach definition of Monad from its concrete implementation like
List or Option, let us define abstract Monad interface using Scala high-order types feature:

```scala
  trait Monad[F[_]] extends Functor[F] {
    def unit[A](a: => A): F[A]

    def flatMap[A,B](ma: F[A])(f: A => F[B]): F[B]

    def map[A,B](ma: F[A])(f: A => B): F[B] = 
      flatMap(ma)(a => unit(f(a))) 
  }

  trait Functor[F[_]] {
    def map[A,B](fa: F[A])(f: A => B): F[B]
  }
```

Functor is one more abstraction which is more simpler than Monad. It requires only `map` implementation. We can say that every Monad also a Functor. 
Functor is also coming from the Category Theory. I decided to mention it here, because you will frequently find it in the context of Monads,
when learning functional programming in general. Abstract Monad interface can also implement map in terms of `flatMap` and `unit` functions, 
so that `map` is implemented automatically for any concrete implementation of some Monad.

## Function application in flatMap

An application of `f` function in `flatMap` and `map` depends on the concrete Monad instance. In one case the lambda
function we pass to the `flatMap` is always executed, in another cases not. Examples:

"f" applied when:

- Option[A]: is Some(A)
- Either[A, B]: is Right(B)
- List[A]: is non-empty
- Future[A]: is ready

Even though `flatMap` behaves differently on concrete Monad instance, there is still great benefits to use them in any ordinary program.
In order to classify some type as a Monad, it needs to comply with **Monad Laws** and that is closing the definition of Monads. Let's look 
at Monad laws before we move further to practical examples.

## Monad Laws

### 1. Identity

Result of a function which creates Monad instance using `unit` is equal to application of this function over already created Monad instance.


Example:

```scala
def f(x: Int): Option[Int] = Some(x)

scala> Some(1).flatMap(f) == f(1)
res0: Boolean = true

scala> f(1) == Some(1).flatMap(f)
res1: Boolean = true
```

Abstract definition of Identity Law:

#### 1.1 Left identity

```scala
def f[A](x: A): Monad[A] = unit(x)

flatMap(unit(x))(f) == f(x) 
```

#### 1.2 Right identity

```scala
f(x) == flatMap(unit(x))(f)
```

### 2. Associative

Application of `f1` and `f2` functions one after another yields the same result as applying them within the first `flatMap`.

Example:

```scala
def f1(a: Int): Option[Int] = Some(a + 1)
def f2(a: Int): Option[Int] = Some(a * 2)

scala> Some(1).flatMap(f1).flatMap(f2)
res0: Option[Int] = Some(4)

scala> Some(1).flatMap(a => f1(a).flatMap(f2))
res1: Option[Int] = Some(4)
```

Abstract definition:

```scala
def f1[A](a: A): Monad[A]
def f2[A](a: A): Monad[A]

if x is a Monad instance,

flatMap(flatMap(x)(f1))(f2) == flatMap(x)(a => flatMap(f1(a))(f2))
```

## Functor Laws

### 1. Identity

Example:

```scala
map(Some(1))(a => a) == Some(1)
```

Abstract definition:

```scala
map(x)(a => a) == x  // the same value returned
```

### 2. Associative

Example:

```scala
val f1 = (n: Int) => n + 1
val f2 = (n: Int) => n * 2

map(map(Some(1))(f1))(f2) // Some(4)
            == 
map(Some(1))(f2 compose f1) // Some(4)
```

Standard Scala function `compose` return a function which applies f1 and then f2 taking the result of the first f1 function.

Abstract definition:

```scala
map(map(x)(f1))(f2) == map(x)(f2 compose f1) 
```

## Application of Monads

Using Monads we can do sequential composition. If we have several values in form of Option, we can sequence them into logic program,
which evaluates next value based on the `flatMap` behaviour of the previous value. 

### Compose Option

```scala
final case class Coffee(name: String)

val isOn = Some(1)
val coffeeName = Some("black")
val makeCoffee = (name: String) => Some(Coffee(name))

for {
  _ <- isOn
  name <- coffeeName
  coffee <- makeCoffee(name)
} yield coffee

scala> Option[Coffee] = Some(Coffee(black))
```

Final result of this program is Some(..) value. However, it could result into None, if one these three values is None.

### Compose Either

The following three functions return Either Monad, so that we can compose them into a sequence.

```scala
case class Cluster(pods: Int)

def validateNamespace(ns: String): Either[String, Unit] = Right(())
def clusterExists(ns: String): Either[Cluster, Unit] = Right(())
def createCluster(ns: String, cluster: Cluster): Either[String, Cluster] = 
  Right(Cluster(cluster.pods))
```

We can compose them in same manner as we have done with **Option** example above:

```scala
val ns = "my-cluster"
for {
   _ <- validateNamespace(ns)
   _ <- clusterExists(ns).left.map(c => 
           s"Cluster with ${c.pods} pods already exists")
   newCluster <- createCluster(ns, Cluster(4))
} yield newCluster
```

From business logic perspective we want to create some hypothetical cluster if namespace is valid and cluster for the given namespace does not exist. 
We implemented errors as `Either.Left` and normal result as `Either.Right`. Interface like `Either` is a popular approach not only in Scala to have some sort of result wrapper
for normal and error results.

Final result value is based on the return values we hardcoded in the given functions:

```scala
scala> Either[String,Cluster] = Right(Cluster(4))
```

Benefits of using Monads is that we do not need to use `if/else` control flow, since we have Monads Laws working when we compose Monad instances.

In case some of the given function returns `Either.Left`, for example:

```scala
def validNamespace(ns: String): Either[String, Unit] = 
   if (ns == "my-cluster") 
   Left(
     “Cluster namespace is not valid name, choose another name”
   ) else Right(())
```   

Then it turns the whole result of the composition into error state, i.e. into `Either.Left`:

```scala
scala> Either[String,Cluster] = Left(
              Cluster namespace is not valid name, choose another name
            )
```            

## For comprehension

Scala offers special syntax for the sequence of nested `flatMap` calls and one `map` at the end, which is called "for-comprehension".

**for {…} yield** is a syntactic sugar for a sequence of calls:

```scala
flatMap1(… + flatMapN(.. + map(…)))
```

**Desugared version**:

Behind the scene, Scala compiler desugars the `for-comprehension` into the following code:

```scala
validNamespace("my-cluster")
  .flatMap(_ =>
     clusterExists(ns)
       .left
       .map(c => s"Cluster with ${c.pods} pods already exists")
       .flatMap(_ =>
            createCluster(ns, Cluster(4))
               .map(newCluster => newCluster)
        )
  )
```

**Sugared version of the same code snippet**:

```scala
for {
  _ <- validNamespace("my-cluster")
  _ <- clusterExists(ns).left.map(c => 
          s"Cluster with ${c.pods} pods already exists")
  newCluster <- createCluster(ns, Cluster(4))
} yield newCluster
```

For-comprehension of this program is much more readable and thus recommended to be used when composing monadic values in particular programs.

## Caveat with Monads

{{ resize_image(path="scala-monads/monads-caveat.png", width=620, height=620, op="fit_width") }}

 We can easily compose Monads of the same types, like we have seen in examples, all values were options or eithers and so on. 
 However, it is not straightforward to compose different Monad stacks, like Option and Either values in one sequence.
 Let's look at the example of such problem below.

### Problem

Let's make one of the value in the `for-comprehension` to be different type, so that we will try to compose different Monads:

```scala
def validateNamespace(ns: String): 
    Either[String, Unit]

def clusterExists(ns: String): 
    Option[Either[String, Cluster]] //Attention <-- two Monad layers

def createCluster(ns: String, cluster: Cluster): 
    Either[String, Cluster] 
```

If we try to compile below code:

```scala
for {
  _ <- validateNamespace(ns)
  cluster <- clusterExists(ns)
  updated <- createCluster(ns, cluster)
} yield  updated
```

This is going to end up in compiler errors:
```scala
updated <- createCluster(ns, cluster)
                             ^
<pastie>:4: error: type mismatch;
 found   : Either[String,Cluster]
 required: Cluster

    cluster <- clusterExists(ns)
            ^
<pastie>:3: error: type mismatch;
 found   : Option[Nothing]
 required: scala.util.Either[?,?]
Option[Nothing] <: scala.util.Either[?,?]?
false
```
__First Monadic value rules them all__. 

Once we put first value such as `validateNamespace`, which returns `Either[_, _]`, it starts
to drive the return type of the `flatMap` function. Second nested value is not `Either` type, but `Option[_]`. Here it starts to brake
the Monad interface and eventually won't let it compile the code. What we need is to align monadic values to common ground.

## Monad Transformer

 In order to compose different Monad types, we can use one more pattern called [Monad Transformer](https://en.wikipedia.org/wiki/Monad_transformer).

 Monad Transformer is a custom-written Monad designed specifically for composition. Of course we could tackle above problem
 by unboxing Option, then checking what is in the Either, return Either again to make `for-comprehension` to be compiled. However, 
 this would be clumsy and not scalable solution in terms of code maintenance. Monad Transformers example:

 - `OptionT` to compose `Option` + Any other Monad
 - `EitherT` to compose `Either` + Any other Monad
 - `ReaderT` to compose `Reader` + Any other Monad
 - `WriterT` to compose `Writer` + Any other Monad
 - ... others

 If we want to compose Option Monad with other monadic values of type `Either`, then we need to use `EitherT` monad for `Option`.
 `EitherT` instance knows how to unbox and box `Option` to operate on nested `Either` and thus guide the `flatMap` function.

 Let us look at `EitherT` example implementation taken from Cats library:

 ```scala
// takes 3 type parameters: 
// 1. high-order type of the outer Monad 
// 2. left type of Either
// 3. right type of Either
final case class EitherT[F[_], A, B](value: F[Either[A, B]]) {

  def flatMap[AA >: A, D](f: B => EitherT[F, AA, D])(implicit F: Monad[F])
    : EitherT[F, AA, D] =
    // Attention: there is one more "flatMap" to unwrap first Monad layer,
    // which is F[_] 
    EitherT(F.flatMap(value) { 
      case l @ Left(_) => F.pure(l.rightCast)
      case Right(b)    => f(b).value
    })
}
```

See inline comments above. One more important point is that we expect an implicit Monad instance for that outer Monad 
`F[_]`. We use it to unwrap first Monad, by convention this variable is also named `F`. 
So Monad Transformer does not do any magic, but it is just a type constructor, which returns a Monad as result. 

### Apply Monad Transformer

Now let us use the same example and define return values:

```scala
case class Cluster(pods: Int, updated: Long)

def validateNamespace(ns: String): Either[String, Unit] = 
  Right(())

def clusterExists(ns: String): Option[Either[String, Cluster]] =
  Some(Right(Cluster(3, System.currentTimeMillis())))

def updateCluster(ns: String, cluster: Cluster): 
  Either[String, Cluster] =
  Right(Cluster(cluster.pods, System.currentTimeMillis()))
```

We are going to use `EitherT` instance from [Cats](https://typelevel.org/cats/datatypes/eithert.html) library.

```scala
import cats.implicits._
import cats.data.EitherT

val cluster = for {
    _ <- validateNamespace(ns).toEitherT[Option]
    cluster<- EitherT(clusterExists(ns))
    updated <- updateCluster(ns, cluster).toEitherT[Option]
} yield  updated
```

Since we introduced Monad transformer into composition, we have to use it for all the monadic values in the same sequence of flatMaps.
So, we have to wrap first value and third value into EitherT as well using extension method `to EitherT`.

In the result we have two layers of Monads too. First `Option`, then `Either`:

```scala
scala> cluster.value
Some(Right(Cluster(3,1583095558496)))
```

Alternative case, when some of the statement in composition yields an error value:

```scala
// we return Left value this time
def clusterExists(ns: String): Option[Either[String, Cluster]] =
    Left("Cluster is invalid").some

scala> cluster.value
res4: Option[Either[String,Cluster]] = Some(Left(Cluster is invalid))
```

In the result, our composition stopped on the second statement. `clusterExists` returns Some(Left(...)), so that 
`EitherT` could detect that `Either.Left` is end of the journey and entire composition ended on `Left` even it is wrapped into `Some`.
Basically, Monad transformer looks into two layers one by one, when chaining monadic values. This was our goal
to get a concise program and handle nested monadic value on composition in the same time.

## Summary

Monad and Monad transformers are useful abstractions in every day life of functional programmer.
Although it may seems like Monad is programming language on its own, it allows us to write programs based on the Laws!!!
We can compose different monadic values without using much a control flow. 
In the result, we get fewer logical errors in the code, better structured programs and 
what is more important we get possibility to change programs in future much easier without breaking the entire world.

Now go and flatMap all the things :-)