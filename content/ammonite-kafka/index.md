+++
title="Ammonite Kafka Producer"
date=2020-11-29
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["kafka", "ammonite"]
categories = ["scala"]
+++

If you need to run a Scala code as a script, i.e. using Scala source file to execute some short code, 
then [Ammonite Scripts](https://ammonite.io/#ScalaScripts)
may be a solution for you. [Ammonite project consists](https://ammonite.io/#Ammonite) of a REPL, script launcher and a few Scala libraries.  
Let’s write a script to generate JSON data and send it to [Apache Kafka topic](https://kafka.apache.org/documentation/#quickstart_createtopic).
<!-- more -->
# Create scripts

We are going to create two scripts. One script will be importing another script definitions. 
In this case will be able to organise our code in modules. Even short scripts may become complex and should be a bit organised for better code maintenance.

## First Script - Generate Data

```bash
touch common.sc
```
This script imports two libraries:

- `scalacheck` for data generation
- `circe` to format case classes to JSON text

```scala
import $ivy.`org.scalacheck::scalacheck:1.14.3`
import $ivy.`io.circe::circe-generic:0.13.0`

import io.circe.generic.auto._, io.circe.syntax._
import org.scalacheck.Gen

case class Sensor(number: Int, lat: Double, lon: Double, address: String)
case class Event(time: Long, temperature: Float, sensor: Sensor)

val sensor1 = Sensor(1, -75.5712, -130.5355, "123 Main St, LAX, CA")
val sensor2 = Sensor(2, -48.8712, -151.6866, "456 Side St, SFO, CA")
val sensors = Gen.oneOf(sensor1, sensor2)
val temperature = Gen.choose(21f, 25f)

def genEvent =
  for {
    s <- sensors
    t <- temperature
  } yield Event(System.currentTimeMillis(), t, s)

def genBatch(size: Int = 10) =
  for {
    events <- Gen.listOfN(size, genEvent)
    dataBatch = events.map(_.asJson.noSpaces)
  } yield dataBatch
```

`genBatch` is main function in this script to be used by another/main script. It generates `List[String]` as return value. 
Every string in the list is a JSON text to be sent to Kafka topic.


## Second Script - Send Data

```bash
touch kafka-producer.sc
```

Add below Scala code statements to import Kafka library, Scala, Java classes into our script

```scala
// file import explained later 
import $file.common
// Ammonite imports library automatically!
import $ivy.`org.apache.kafka:kafka-clients:1.0.0`

import common._
import org.apache.kafka.clients.producer.Producer
import org.apache.kafka.clients.producer.ProducerRecord
import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.producer.ProducerConfig
import org.apache.kafka.common.serialization.LongSerializer
import org.apache.kafka.common.serialization.StringSerializer
import java.util.Properties
import java.io.FileReader
import java.io.File
import scala.annotation.tailrec
import scala.util.Using
import scala.io.Source
```

In order to make this script executable we need to annotate one of the method by `@main` annotation:

```scala
@main
def main(topicName: String, delay: Int = 1000) = {
  // create producer properties
  val configFile = new File("./producer.properties")
  val props = new Properties()
  Using.resource(new FileReader(configFile)) { reader =>
    props.load(reader)
  }
  val bootstrapFile = "bootstrap-servers.txt"
  val servers = Using.resource(Source.fromFile(bootstrapFile)) {
    file =>
      file.getLines.toList.headOption
        .getOrElse(sys.error(s"$bootstrapFile file is empty!"))
        .trim
  }
  props.put(
    "bootstrap.servers",
    servers
  )
  props.put(ProducerConfig.CLIENT_ID_CONFIG, "KafkaExampleProducer")
  props.put(
    ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
    classOf[LongSerializer].getName
  )
  props.put(
    ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
    classOf[StringSerializer].getName
  )
  props.put(ProducerConfig.ACKS_CONFIG, "all");
 
  // create producer
  val producer = new KafkaProducer[Long, String](props)
 
  // send data to Kafka
  sendData(producer, topicName, delay)
}
```

In the main function we use regular Scala code to:
- read producer properties
- read Kafka broker/bootstrap hostnames
- create `KafkaProducer` using Java Kafka library API
- send data to Kafka topic infinitely 

Above main function can be executed now via:

```bash
amm kafka-producer.sc --topicName test --delay 1000
```

However, it won't compile until we add `sendData` function.

```scala
@tailrec
def sendData(
    producer: Producer[Long, String],
    topic: String,
    delay: Int
): Unit = {
  genBatch().sample.get.foreach { json =>
    println(json)
    val record = new ProducerRecord(topic, System.currentTimeMillis(), json)
    producer.send(
      record,
      (metadata, ex) => {
        Option(ex).foreach { t =>
          throw new RuntimeException(s"Failed to send the data: $json", t)
        }
      }
    )
  }
  Thread.sleep(delay)
  sendData(producer, topic, delay)
}
```

`sendData` is calling `genBatch()` function from the first script `common.sc`. We imported all definitions from that script via `import common._`.
Ammonite script launcher compiles all imported scripts automatically, so we need to import it only.

In the `sendData` function we generate List of JSON strings to send them via Kafka producer. In case we get an exception in producer's response, we
throw an exception to stop the script execution. In case there is no exception thrown by the Kafka producer, then `sendData` is 
sending data to Kafka topic again and again sleeping in between with the user given delay in milliseconds.

# Execute Script

Before we execute above script, we need to have two configuration files for Kafka Producer.

```bash
touch producer.properties
```

For example, I have the following properties. In your case, you will have some of them or none of them (if you rely on default Kafka
producer properties):

```properties
# this file content is auto-generated by get-jks.sh
ssl.endpoint.identification.algorithm=
ssl.truststore.location=./ssl/producer.truststore.jks
ssl.truststore.password=<your trustore password if any>
ssl.keystore.location=./ssl/producer.keystore.jks
ssl.keystore.password=<your key password if any>
ssl.key.password=<your key password if any>
sasl.mechanism=GSSAPI
sasl.kerberos.service.name=kafka
security.protocol=SASL_SSL
sasl.jaas.config=com.sun.security.auth.module.Krb5LoginModule required \
    useKeyTab=true \
    storeKey=true \
    useTicketCache=false \
    keyTab="./client.keytab" \
    principal="my-test@EXAMPLE.COM";

```

Another file that we used in the main script is file with Kafka broker hostnames. Put your broker hostname(s) or IP address(es) separaing by coma:

```txt
<hostname of the broker 1>,<hostname of the broker 2>....
```
Finally, if we run the main script, it will start to print the JSON data that is sent to Kafka topic:

```bash
amm kafka-producer.sc --topicName test --delay 1000
Compiling ~/kafka-produce.sc
{"time":1606601101634,"temperature":21.944403,"sensor":{"number":2,"lat":-48.8712,"lon":-151.6866,"address":"456 Side St, SFO, CA"}}
{"time":1606601101635,"temperature":24.016523,"sensor":{"number":1,"lat":-75.5712,"lon":-130.5355,"address":"123 Main St, LAX, CA"}}
{"time":1606601101635,"temperature":24.302032,"sensor":{"number":2,"lat":-48.8712,"lon":-151.6866,"address":"456 Side St, SFO, CA"}}
{"time":1606601101635,"temperature":24.295887,"sensor":{"number":2,"lat":-48.8712,"lon":-151.6866,"address":"456 Side St, SFO, CA"}}
{"time":1606601101635,"temperature":24.507029,"sensor":{"number":2,"lat":-48.8712,"lon":-151.6866,"address":"456 Side St, SFO, CA"}}
{"time":1606601101636,"temperature":24.488947,"sensor":{"number":1,"lat":-75.5712,"lon":-130.5355,"address":"123 Main St, LAX, CA"}}
.....
```


# Summary 

If you want to use Scala for scripting purposes, you can easily do that with Ammonite Scripts.
It allows you to import any JVM library from [Sonatype public repository](https://search.maven.org/). 
[Ammonite can be also configured]((https://ammonite.io/#import$ivy)) to search any other public or private Maven, Ivy repositories. 

Another cool feature of Ammonite Scripts is local [file imports](https://ammonite.io/#import$file). That allows to write many scripts and organise them in modules, reuse them in many places.
Code editor like VSCode and IntelliJ IDE support Ammonite Scripts well, so that you can benefit from auto-completion and "go to definition" features while writing your code 
with Ammonite Scripts.

[Standard Scala SDK can also execute Scala file](https://www.artima.com/pins1ed/scala-scripts-on-unix-and-windows.html) `.scala` as shell script. 
One needs to add a special directive at the top of the file, so that the file will be executed via Scala compiler. However, in this way, you can't download some library like Ammonite does.
