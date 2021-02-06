+++
title="Algorithms: Largest Sum Contiguous Subarray"
date=2020-08-13
toc_enabled = false
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["algorithms"]
categories = ["scala"]
+++

Most of the algorithmic tasks are related to iterating over arrays of data. They often can be expressed as a function which takes some input and returns some single value or an array of values. For instance:

```scala
def maxSum(a: Array[Int]): Array[Int] = ???
```
<!-- more -->

Let's solve next task using Scala's elegant and concise syntax.

### Find the max sum in the given array

It is one of the interesting task that I met first time at one of my past job 5 years ago. It was an internal meetup of developers learning new languages as well as training themselves in programming for fun.

Later I use this task in Scala pet-project, called [DevGym](https://github.com/novakov-alexey/devgym). Project idea was similar to well-known Codility web-site.


### Task

There is one array as input and another array as output. See test examples below:

```scala
// Given
Array(2, -4, 2, -1, 3, -3, 10, -1, -11, -100, 8, -1) 

// Then
       Array(2, -1, 3, -3, 10)
```

2 - 1 + 3 - 3 + 10 = 11

Since `11` is the max sum we could find within the input array

```scala
//Given
Array(-2, 1, -3, 4, -1, 2, 1, -5, 4)
//Then
           Array(4, -1, 2, 1)
```

4 - 1 + 2 + 1 = 6

`6` is the max sum.

### Solution

In general, we can solve this problem in O(n) complexity. By using intermediate variables to 
accumulate current sum as well as max sum, we can find `left` and `right` indices, which can be used
to return a sub-array.


```scala
def maxSum(a: Array[Int]): Array[Int] = {
  var currentSum = 0
  var maxSum = 0
  var left, right, maxLeft, maxRight = 0
  var maxI = 0 //used when all negatives in the array

  for (i <- a.indices) {
    currentSum += a(i)

    if (currentSum > 0) {    
      // in case current sum is getting greater,
      // then we found next right index with the max sum so far
      if (currentSum > maxSum) {        
        maxSum = currentSum
        right = i
        maxRight = right
        maxLeft = left
      }
    } else {
      // in case new sum is lower than or equal 0, 
      // then we need to move left and right index further  
      // and continue the search
      left = i + 1
      right = left
      currentSum = 0
      if (a(i) > a(maxI)) maxI = i
    }
  }
  // at this point we found left and right
  // indices to capture sub-array with max sum
  if (maxLeft == a.length) Array(a(maxI))
  else a.slice(maxLeft, maxRight + 1)
}
```

Above algorithm can be even shorter, if we return only final sum, i.e. without tracking left and right indices.

# Summary

Scala is a nice language to solve different algorithmic tasks. Try your next project in Scala, you will be surprised how fun to write code in it. 

See my other blog-posts for algorithms in Scala:

1. [Dijkstra shortest path](https://medium.com/se-notes-by-alexey-novakov/algorithms-in-scala-dijkstra-shortest-path-78c4291dd8ab)
2. [Rolling Sum](https://medium.com/se-notes-by-alexey-novakov/rolling-sum-in-scala-6bc9a5a82e75)