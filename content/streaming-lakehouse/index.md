+++
title="Iceberg and Paimon as building blocks of Streaming Lakehouse"
date=2026-05-09
draft = true

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["big data", "flink", "paimon", "iceberg"]
categories = ["architecture"]
+++

This blog will uncover relatively new Data Platform architectture called Streaming Lakehouse or Streamhouse for short.
We will look back at the history of Data Lakes and why they appeared and how they evolved into Lakehouses. Main part will be focusing on technilogies why Streamhouse architecture such Apache Flink and Paimon. Let's get started!

# Data Warehouse, Data Lake, Lakehouse, Streamhouse, any other house?

Before the Big Data space was created, many companies kept their historical data in data warehouses. Data warehouse was usually a licensed database software like Oracle, SQL Server, Teradara and others. Although, these databases were mainly focusing to store transactional data, they could already support relatively big volumes to store and process. Soon, the internet companies like Yahoo realised that data warehouse efficiency was not enough to aggregate more data which they started to accumulate in early 2000s.
People saw that they need to store terabytes of data daily and be able to process it on cheap comodity hardware.

Eventually, Hadoop project was born. Hadoop consists of a couple of components such as Map-Reduce, HDFS, Hadoop-common and YARN. Most of the time when people say Hadoop they refer to all its components. Today, in 2026 we still see that the Hadoop project makes new releases, so that data teams can upgrade their already legacy environments. There is already a old trend to replace Hadoop installation with something like Lakehouse, which we describe later. Most existing data management frameworks, especially from Apache Foundation still support HDFS asone of the files system for data processing. Another example, the hadoop-common library "leaked" to S3 integration which led to long-lasting dependency of Hadoop in already modern frameworks, which is a bit unfortunate. It is really hard to get rid of Hadoop project dependencies. However, many OSS project in data management space already started to create new wave of libraries to detach themselves from Hadoop. 

A Data Lake architecture is an idea to process all kind of database wihtin one platform like Hadoop. Engineers could store structure, semi-structure and unstructured data in HDFS. They were writing Map-Reduce jobs in Java language to read data in whatever format/structure it was stored on HDFS and to process it in parallel using Hadoop cluster workers. Computation results were stored back to HDFS. Such approach inspired some people to call this architecture as Data Lake. Lake comes from idea that engineers could store data in any shape and define schema when reading, i.e. they could work with data without those typical constraints that traditional databases and data warehouse implied. 

Althogh Data Lakes was a step towards "Not Only SQL" paradigm, people quickly realised that they are missing those useful tools from RDMS world like SQL, tables, database schemas, indicies, table statistics, changes as committs to metadata layer, ACID transactions, time traveling and data rollbacks. 




