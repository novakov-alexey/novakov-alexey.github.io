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

Current version of Apache Flink 1.16 still depends on Scala 2.12. This Scala version will probably be tied to Flink for a while taking into account how slow big open-source projects are migrating from
one Scala version to another. 

<!-- more -->

{{ resize_image(path="flink-scala3/images/flink-github-scala-version.png", width=700, height=600, op="fit_width") }}

For those who wants to use Scala 2.13 or [Scala 3](https://docs.scala-lang.org/scala3/new-in-scala3.html) already today, there are few options available: 

- [flink-scala-api](https://github.com/findify/flink-scala-api) a fork of Flink Scala bindings from Findify 
- [Flink4s](https://github.com/ariskk/flink4s)

Below example is using first option - [flink-scala-api](https://github.com/findify/flink-scala-api).

We are going to create custom Flink Docker image which can run Scala 3 streaming jobs based on Flink 15.2 version.

# Build Job with Scala3 embedded

Clone [novakov-alexey/flink-sandbox](https://github.com/novakov-alexey/flink-sandbox) repository which is using SBT to build an example Flink job.
Package Scala code into self-sufficient (fat) JAR file:

```bash
sbt assembly
```

After running SBT `assembly` task, you get a JAR file in the target folder. Checking:

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
RUN rm $FLINK_HOME/opt/flink-cep-scala_2.12-1.15.2.jar
RUN rm $FLINK_HOME/lib/flink-scala_2.12-1.15.2.jar

RUN mkdir -p $FLINK_HOME/usrlib
COPY ./target/scala-3.1.2/flink-sandbox-*.jar $FLINK_HOME/usrlib/my-flink-job.jar
```

Now let's build an image:

```bash
docker build -t flink:1.15.2-my-job-scala3 .
```

# Launch Flink Job

Install Flink on the client machine to get Flink CLI around. Check:

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

Since Flink is not tightly coupled to Scala already, one can remove Scala dependency to Scala 2.12 to plug in newer version.
One can either add own Scala version as part of Docker image build or just pack it inside the Flink job JAR.

Hopefully we will see cleaner solution for support of Scala 2.13 and 3 in Apache Flink in future.