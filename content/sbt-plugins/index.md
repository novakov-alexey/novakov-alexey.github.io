+++
title="SBT Plugins"
date=2020-06-29
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["sbt", "plugins"]
categories = ["scala"]
+++

SBT is a Scala Build Tool. It is written in Scala and can compile, build artefacts for Scala and Java projects. SBT is also the first build
tool in the Scala eco-system and the most used one among Scala developers. I am using SBT already for many years and found the following useful plugins which I use in most of my projects:
<!-- more -->
## Giter8

It allows to create an SBT project template to ramp up new SBT project. Project template usually includes typical configuration that 
an SBT user copies and pastes from a project to project. User can put any file into a template.

One part of the Giter8 is embedded into SBT. The second part is the Giter8 plugin itself. User can create new SBT project from template hosted at GitHub and that the most useful part. You just need an internet access, then run SBT `new` command. For example:

```scala
sbt new novakov-alexey/scalaboot.g8     
```

scalaboot.g8 is a name of the GitHub repository at my personal account `novakov-alexey`. SBT converts template name into  
https://github.com/novakov-alexey/scalaboot.g8 path, which you can visit in browser as well. There is also an option to use Giter8 template from the local file system.

Once we run above command, Giter8 creates new project file structure such as:

```bash
├── build.sbt
├── project
│   ├── Dependencies.scala
│   ├── build.properties
│   └── plugins.sbt
├── src
│   ├── main
│   └── test
└── version.sbt
```

### Template File Structure

Project template is also an SBT project :-). In case of scalaboot.g8 example, it looks like this:

{{ resize_image(path="sbt-plugins/scalaboot.png", width=600, height=600, op="fit") }}

Content of the g8 folder is a template content which will be used when users apply this template
for their projects. Giter8 template supports properties and special syntax for them. Properties can
be put at any place of your template. Let's look at the example in build.sbt file:

```scala
ThisBuild / organization := "com.example"
ThisBuild / scalaVersion := "$scalaVersion$"
ThisBuild / description  := "$desc$"

lazy val root = (project in file(".")).
  settings(
    name := "$name;format="lower,hyphen"$",
    libraryDependencies ++= Seq(
      akkaHttp,
      akkaStreams,
      scalaLogging,
      logback,
      …

```

A text between `$ ... $` is evaluated by Giter8 and replaced by pre-defined or user-given parameters.
In my `scalaboot` template, I have following pre-defined values:

*src/main/g8/default.properties*

```properties
name=Project Name
desc=Describe your project a bit
scalaVersion=2.13.1
```

SBT `new` command is going through the list of defined properties and sets the default values in case
users do not provide their own values. In the example below, I apply `scalaboot` template for
project name `app-1`. I also set my project description and newer Scala version. They will appear in the build.sbt:

```bash
[info] Loading settings for project global-plugins from idea.sbt,gpg.sbt ...
[info] Loading global plugins from /Users/alexey/.sbt/1.0/plugins
[info] Set current project to git (in build file:/Users/alexey/dev/git/)
name [Project Name]: app-1
desc [Describe your project a bit]: order registration
scalaVersion [2.13.1]: 2.13.2

Template applied in /Users/alexey/dev/git/./app-1
```

## SBT Revolver

It is simple plugin to be added in a Scala project. However, Giter8 is not necessarily to be added in your project, only when you
develop new template and want to keep it in the GitHub repo. Revolver plugin can be added to project as any other user plugin via project/plugins.sbt file.
Adding its definition to that file:

```scala
addSbtPlugin("io.spray" % "sbt-revolver" % “x.y.z")
```

Change x.y.z to latest version from its GitHub repository. 

Main feature of sbt-reolver is "triggered execution" upon project file modification. 
It helps to restart your application automatically and may remind
you dynamic-language experience where developers test their modules by refreshing a browser page or by calling their scripts again.

{{ resize_image(path="sbt-plugins/fe-dev.png", width=600, height=600, op="fit") }}

In summary, *it enables a super-fast development turnaround for your Scala applications*.

SBT Reolver starts your application in forked JVM, that helps to easily pass JVM options and restart it again upon triggered execution.

{{ resize_image(path="sbt-plugins/revolver-jvm.png", width=600, height=600, op="fit") }}

*MainExchange* is my Scala application started by sbt-revolver as separate JVM process.

Revolver has its own configuration to control JVM options, environment variables, etc.

JVM options example:
```scala
javaOptions in reStart += "-Xmx2g"
```

Set your main class. It is useful when you have more than one:
```scala
mainClass in reStart := Some("com.example.Main")
```

Enable debug of the forked JVM process:
```scala
Revolver.enableDebugging(port = 5050, suspend = true)
```

Export environment variables for your Scala application:
```scala
envVars in reStart := Map(“K8S_NAMESPACE" -> “test")
```

Below is an example of starting an application via sbt-revolver:

{{ resize_image(path="sbt-plugins/revolver-start.png", width=600, height=600, op="fit") }}

MainExchange is Akka-HTTP based application running HTTP server. Now let us change some line of code in the code base.
Once we done that, sbt-revolver immediately triggers compilation, stop running process and starts new one:

{{ resize_image(path="sbt-plugins/revolver-triggered.png", width=600, height=600, op="fit") }}

There are 3 things happened:

- Build triggered (compilation)
- Stop running application
- Start new application

`restart` revolver SBT task is leveraging SBT triggered execution which is enabled by tilde *~* in front the task name,
 when running it in SBT shell.

 There are other useful commands to be combined with ~ to trigger some task upon files modification:

 ```bash
// runs failed tests, if any
~ testQuick

// runs specific test
~ testOnly org.alexeyn.SomeTest

// runs all tests
~ test

// cleans compiled sources and runs all tests
~ clean; test
 ```

 ## SBT Tpolecat

 Enables Scala compiler options as per recommendations of Rob Norris [blog-post](https://tpolecat.github.io/2017/04/25/scalac-flags.html). Plugin enables as many Scala compiler options as possible to enforce type safety and
 discourage bad practises in the code base by turning warnings into compiler errors.

Add plugin to your project:

 ```scala
 addSbtPlugin(“io.github.davidgregory084" % "sbt-tpolecat" % “0.1.10")
 ```

 Actually, the same compiler options can be enabled manually within the SBT definition. However, it is more convenient to
 enable this plugin once and forget about adding anything manually. One can also disable particular options enabled by this
 plugin, in case that option does not make sense for particular project.

 Some of the options which are enabled by this plugin:

 ```scala
 scalacOptions ++= Seq(
  "-deprecation",               
  "-encoding", "utf-8",         
  "-explaintypes",                  
  "-language:higherKinds",          
  "-language:implicitConversions",  
  "-unchecked",                       
  "-Xfatal-warnings",            
  "-Xlint:infer-any",                 
  "-Ywarn-dead-code",              
  "-Ywarn-extra-implicit",        
  "-Ywarn-inaccessible",          
  "-Ywarn-infer-any",            
  "-Ywarn-numeric-widen",       
  "-Ywarn-unused:implicits",   
  "-Ywarn-unused:imports",     
  "-Ywarn-unused:locals",     
  "-Ywarn-unused:params",     
  "-Ywarn-unused:patvars",    
  "-Ywarn-unused:privates",   
  "-Ywarn-value-discard"     
…    
)
 ```

Last time I checked this plugin it enables 54 scalac options. I recommend to use this plugin by default in every project, it will
make your code base much more robust.

## SBT Native Packager

To enbale in your project:

```scala
addSbtPlugin("com.typesafe.sbt" %% "sbt-native-packager" % “x.y.z")
```

Native Packager allows to package your application in different formats such as:
- universal zip, tar.gz, xz archives
- deb and rpm packages
- dmg 
- msi 
- Docker
- GraalVM native images

Native packager is not auto-plugin, i.e. it is not enabled by default. In order to use it for some of
your module, you need to enable it in SBT definition:

```scala
lazy val root = (project in file(".")).
  settings(
    name := "exchange",
    ….
    dockerBaseImage := “openjdk:8-jre-alpine”,
    dockerExposedPorts ++= Seq(8080),
    dockerRepository := Some(“alexeyn")
  ).enablePlugins(AshScriptPlugin) 
                          // or other options - DockerPlugin, JavaAppPackaging
```
This plugins comes with different types of packaging format, which you can choose when enabling it for some SBT module.
In the example above, we enable Java packaging format with Ash shell compatible executable script, so that we can run a JAR file
in Alpine Linux. Basically, JavaAppPackaging is a base format. It creats a couple of scripts to start JVM with a long list 
of JAR files in the CLASSPATH variable. It also puts all required dependencies into the `lib` folder, 
which is referenced from that automatically generated shell script. 

### Java Packaging Format

Below an example of such SBT task. It builds a universal ZIP archive:

```scala
sbt universal:packageBin
```

it will create a ZIP archive with a file structure shown below:

```bash
~/dev/git/exchange/target/universal/exchange-0.1.1-SNAPSHOT.zip

tree -L 2
├── bin
│   ├── exchange
│   └── exchange.bat
└── lib
    ├── ch.qos.logback.logback-classic-1.2.3.jar
    ├── ch.qos.logback.logback-core-1.2.3.jar
    ├── com.chuusai.shapeless_2.13-2.3.3.jar
    ├── com.example.exchange-0.1.1-SNAPSHOT.jar
    ├── com.google.protobuf.protobuf-java-3.10.0.jar
    ├── com.typesafe.akka.akka-actor_2.13-2.6.1.jar
    ├── com.typesafe.akka.akka-http-core_2.13-10.1.11.jar
    ├── com.typesafe.akka.akka-http_2.13-10.1.11.jar
    ├── com.typesafe.akka.akka-parsing_2.13-10.1.11.jar
    ├── com.typesafe.akka.akka-protobuf-v3_2.13-2.6.1.jar
    ├── com.typesafe.akka.akka-stream_2.13-2.6.1.jar
    ├── com.typesafe.config-1.4.0.jar
    ├── com.typesafe.scala-logging.scala-logging_2.13-3.9.2.jar
    ├── com.typesafe.ssl-config-core_2.13-0.4.1.jar
    ├── de.heikoseeberger.akka-http-circe_2.13-1.30.0.jar
    ├── io.circe.circe-core_2.13-0.12.3.jar
    ├── io.circe.circe-generic_2.13-0.12.3.jar
    ├── io.circe.circe-jawn_2.13-0.12.3.jar
    ├── io.circe.circe-numbers_2.13-0.12.3.jar
    ├── io.circe.circe-parser_2.13-0.12.3.jar
    ├── org.reactivestreams.reactive-streams-1.0.3.jar
    ├── org.scala-lang.modules.scala-java8-compat_2.13-0.9.0.jar
    ├── org.scala-lang.modules.scala-parser-combinators_2.13-1.1.2.jar
    ├── org.scala-lang.scala-library-2.13.1.jar
    ├── org.scala-lang.scala-reflect-2.13.1.jar
    ├── org.slf4j.slf4j-api-1.7.26.jar
    ├── org.typelevel.cats-core_2.13-2.0.0.jar
    ├── org.typelevel.cats-kernel_2.13-2.0.0.jar
    ├── org.typelevel.cats-macros_2.13-2.0.0.jar
    └── org.typelevel.jawn-parser_2.13-0.14.2.jar

2 directories, 32 files
```

*bin/exchange* is a shell script to run your Scala application Main class.

### Docker Image format

SBT task to create a Dockerfile and the same file structure as for Java packaging format:

```scala
sbt docker:stage
```

```bash
cd /Users/alexey/dev/git/exchange/target/docker
tree -L 5
.
└── stage
    ├── Dockerfile
    └── opt
        └── docker
            ├── bin
            │   ├── exchange
            │   └── exchange.bat
            └── lib
                ├── ch.qos.logback.logback-classic-1.2.3.jar
                ├── ch.qos.logback.logback-core-1.2.3.jar
                ├── com.chuusai.shapeless_2.13-2.3.3.jar
                ├── com.example.exchange-0.1.1-SNAPSHOT.jar
```                

In order to build an image and publish it to a container registry:

```scala
sbt docker:publish
```

You can also customise Dockerfile, which is by default generated automatically. 
Default docker file content can be be seen via:

```scala
sbt> show dockerCommands

[info] * Cmd(FROM,WrappedArray(openjdk:8, as, stage0))
[info] * Cmd(LABEL,WrappedArray(snp-multi-stage="intermediate"))
[info] * Cmd(LABEL,WrappedArray(snp-multi-stage-id="b8437d6f-af0a-459c-ae51-cd3b9c5b7404"))
[info] * Cmd(WORKDIR,WrappedArray(/opt/docker))
[info] * Cmd(COPY,WrappedArray(opt /opt))
[info] * Cmd(USER,WrappedArray(root))
[info] * ExecCmd(RUN,List(chmod, -R, u=rX,g=rX, /opt/docker))
[info] * ExecCmd(RUN,List(chmod, u+x,g+x, /opt/docker/bin/exchange))
```

In order to customize Dockerfile content you can set your sequence of Dockerfile commands:

```scala
dockerCommands := Seq(
  Cmd("FROM", "openjdk:8"),
  Cmd("LABEL", s"""MAINTAINER="${maintainer.value}""""),
  ExecCmd("CMD", "echo", "Hello, World from Docker")
)
```

Sometimes I use only Java packaging part of this plugin and build Docker image directly via `docker build` command
to avoid new SBT start. It is usually done, when image build is designed as separate step in CI pipeline.

## SBT Release

SBT Release provides customisable release process. It helps to manage your project version, publish project artefacts to configured 
repository. 

```scala
addSbtPlugin("com.github.gseitz" % "sbt-release" % “1.0.12")
```

Typical Scala project release process may include:

{{ resize_image(path="sbt-plugins/release-process.png", width=600, height=600, op="fit") }}

You can script all typical tasks to perform version increase, creating Git tag, building an image, publishing
a JAR file to central artefact repository, etc. SBT Release gives a list of predefined tasks, which we
can use as is or customise to fulfil project needs.

Default list of release steps is:

```scala
releaseProcess := Seq[ReleaseStep](
    checkSnapshotDependencies,
    inquireVersions,
    runTest,
    setReleaseVersion,
    commitReleaseVersion,
    tagRelease, 
    publishArtifacts,
    inquireVersions, 
    setNextVersion, 
    commitNextVersion, 
    pushChanges  
)
```

Default list can be good enough for typical Scala project. You do not need to define it in SBT build file if you are fine
with it.

In order to run SBT tasks to release with default steps, one can run:

```scala
sbt 'release with-defaults'
```

Some of the steps are responsible for project version management. Project version is usually located in project root folder and named
as version.sbt file. 

In case we have such version in the file: 

```scala
version in ThisBuild := "0.1.1-SNAPSHOT"
```
then
- inquireVersions step will read it 
- setReleaseVersion step will make as release version

```scala
version in ThisBuild := “0.1.1"
```
- setNextVersion step will switch release version to next snapshot version

```scala
version in ThisBuild := “0.1.2-SNAPSHOT"
```

Version increment can be customised, so that you can control which number is incremented: patch, minor or major version.

There are steps to commit and push changes, typically placed at the end of the process. 
If we look at Git log after release is executed, then we will see that sbt-release is making a couple of commits to reflect
the release process in Git commit history.

Latest message on top:

```bash
commit 99b1094dce14bf99b6f38a8ff9870edaf7c728d3 (HEAD -> master, origin/master)
Date:   Fri Feb 7 09:20:03 2020 +0100

    Setting version to 0.1.2-SNAPSHOT

commit cb9ec293a11a5f6d989c936b18922d3f3ec40bcd (tag: v0.2.2)
Date:   Fri Feb 7 09:16:59 2020 +0100

    Setting version to 0.1.1

commit 63abea7141901419ad732d354dc703f884e53010
Merge: b180810 a1c0c14
Date:   Fri Feb 7 08:57:00 2020 +0100

    Merge pull request #35 from novakov-alexey/add-cookier-attributes
    add string property attributes to put user defined parameters into th…
```

## Other useful plugins

There are many other useful plugins I usually use from project to project:

- *sbt-updates* to report newest versions of the libraries inlucluded in your project.
- *sbt-scalafmt* to format Scala code.
- *sbt-mdoc* to compile Scala code snippets in the Markdown documentation. Useful for own Scala libraries.
- *sbt-scoverage* to report test coverage in the project.

## Summary

SBT has become quite mature build tool. It offers good variety of plugins for every day life of Scala developer.
If you cannot find specific SBT plugin that fits your requirements, you can try to implement it using SBT Tasks and Plugin API. 
Then it could be published as open source project. This is how many SBT plugins were born.