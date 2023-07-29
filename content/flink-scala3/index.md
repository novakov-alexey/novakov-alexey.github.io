+++
title="Using Scala 3 with Apache Flink"
date=2023-01-16
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["big data", "streaming data", "scala3"]
categories = ["scala", "flink"]
+++
<style>
  .container {    
    justify-content: center;
  }
</style> 
<div class="container">
{{ resize_image_no_br(path="flink-scala3/images/flink_squirrel_500.png", width=300, height=300, op="fit_width") }}
{{ resize_image_no_br(path="flink-scala3/images/scala-logo.png", width=300, height=300, op="fit_width") }}
</div>
<br/>

If you have come here,  then you probably know that current version of Apache Flink 1.16 Scala API still depends 
on Scala 2.12. However, the good news is that previous Flink release 1.15 introduced [an important change](https://flink.apache.org/2022/02/22/scala-free.html)
for Scala users, which allows us to use own Scala version via Flink Java API. That means users can now use Scala 2.13 or 3 to write Flink jobs.
Earlier, Flink had Scala 2.12 as a mandatory library on application classpath, so one could not use newer version due to version conflicts.

{{ resize_image(path="flink-scala3/images/flink-github-scala-version.png", width=700, height=600, op="fit_width") }}

Official Flink Scala API will be still available in future, but will probably be deprecated at some point. It is unclear at this point. 
Flink community made a decision to not engage with newer Scala version and make Flink to be Scala-free, in terms of user's Scala version choice.
Whether it is good or bad for Scala users in general we will going to see in near future. Definitely this choice is good, as it unlocks 
Flink for newer Scala versions.

<!-- more -->



For those who wants to use Scala 2.13 or [Scala 3](https://docs.scala-lang.org/scala3/new-in-scala3.html) with Flink already today, there are few options available: 

- [flink-scala-api](https://github.com/flink-extended/flink-scala-api) a fork of Flink Scala bindings originally done by Findify
- [Flink4s](https://github.com/ariskk/flink4s)

Further example in this blog-post is using first option, i.e. [flink-scala-api](https://github.com/flink-extended/flink-scala-api).

# Build Job with Scala 3

We are going to create custom Flink Docker image, which can run Scala 3 streaming jobs based on Flink 15.2 version.
Docker and containers are used to easily reproduce this example on any environment, however they are not mandatory for Scala 3 usage in general.

First, clone [novakov-alexey/flink-sandbox](https://github.com/novakov-alexey/flink-sandbox) Git repository, which is using SBT to build an example Flink job.
So SBT needs to be installed as well. Then package Scala code into self-sufficient (fat) JAR file:

```bash
sbt assembly
```

After running SBT `assembly` task, you get a JAR file in the target folder:

```bash
ls target/scala*/*jar

target/scala-3.1.2/flink-sandbox-assembly-0.1.jar
```

# Build Image with Job included

Make a Dockerfile with the following content:

```docker
# Dockerfile

FROM flink:1.15.2

# removing tied Scala 2.12 from the Apache Flink image 
RUN rm $FLINK_HOME/lib/flink-scala_2.12-1.15.2.jar

RUN mkdir -p $FLINK_HOME/usrlib
COPY ./target/scala-3.1.2/flink-sandbox-*.jar $FLINK_HOME/usrlib/my-flink-job.jar
```

Now let's build a Docker image:

```bash
docker build -t flink:1.15.2-my-job-scala3 .
```

# Launch Flink Job

Install Flink on the client machine to get Flink CLI around. After Flink installation let's check its version, it should be 1.15.2

```bash
flink --version

Version: 1.15.2, Commit ID: 69e8126
```

Get Kubernetes cluster available in the current shell:

```bash
kubectl get nodes

NAME                   STATUS   ROLES                  AGE   VERSION
lima-rancher-desktop   Ready    control-plane,master   15d   v1.23.15+k3s1
```

Finally, launch a job:

```bash
flink run-application -p 3 -t kubernetes-application \
  -c org.example.fraud.FraudDetectionJob \
  -Dtaskmanager.numberOfTaskSlots=2 \
  -Dkubernetes.rest-service.exposed.type=NodePort \
  -Dkubernetes.cluster-id=fraud-detection \
  -Dkubernetes.container.image=flink:1.15.2-my-job-scala3 \
  -Dkubernetes.service-account=flink-service-account \
  local:///opt/flink/usrlib/my-flink-job.jar
```

Checking the result:

```bash 
kubectl get pod -l app=fraud-detection

NAME                               READY   STATUS    RESTARTS   AGE
fraud-detection-656b6fcc9c-lhqpl   1/1     Running   0          31m
fraud-detection-taskmanager-1-2    1/1     Running   0          31m
fraud-detection-taskmanager-1-1    1/1     Running   0          31m
```

# Summary

Starting Flink 1.15, it is no longer tightly coupled to Scala, so that one can remove Scala dependency to Scala 2.12 and use newer version.
Depends on situation, user can add own Scala version as part of Docker image build or just pack it inside the Flink job fat-JAR file.

How long the official Flink Scala API with version of Scala 2.12 will remain available in future releases? And what will happen with it.
In order to answer this question you should keep an eye on [Flink communication channels](https://flink.apache.org/community.html)
 like Slack, Jira and mailing lists.

 Even it is going to be removed, users can safely use [flink-scala-api](https://github.com/flink-extended/flink-scala-api) from Flink Extended
 organization. As one of the maintainer, I can guarantee it will be up to date in the future.

# Links

[FLINK-13414 - Add Support for Scala 2.13](https://issues.apache.org/jira/browse/FLINK-13414)