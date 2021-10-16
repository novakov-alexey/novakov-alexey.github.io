+++
title="Kubernetes Operator in Scala for Kerberos Keytab Management"
date=2021-10-16
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["kerberos", "kubernetes", "operator", "devops"]
categories = ["scala"]
+++


<style>
  .container {
    display: flex;
    justify-content: center;
  }
</style> 
<div  class="container">
{{ resize_image_no_br(path="k8s-operator/images/cats-effect-logo.png", width=100, height=100, op="fill") }}
{{ resize_image_no_br(path="k8s-operator/images/k8s-logo.png", width=250, height=100, op="fit") }}
{{ resize_image_no_br(path="k8s-operator/images/scala-logo.png", width=100, height=100, op="fit") }}
</div><br/>


Kubernetes has built-in controllers to handle its native resource such as
- Pod
- Service
- Deployment
- etc. 

What if you want a completely new resource type which would describe some new abstraction in clear and concise way? Such new resource would describe everything in 
one single type which would require 5-10 separate native Kubernetes resources.
<!-- more -->
Good news that such custom resource creation is possible in Kubernetes, however who would be processing such custom resources?
The answer is the `Custom Controller` would process such new custom resources and this is official way that Kubernetes allows its users
to extend Kubernetes API-Server:

1. kind: CustomResourceDefinition, apiVersion: apiextensions.k8s.io/v1
2. Custom Controller, which is your program running usually inside the Kubernetes cluster 

# Operator Pattern

Custom Controller is also called `Kubernetes Operator`. That also became a design pattern to automate new application deployment operations by
running custom program which will deploy, redeploy native Kubernetes resources based on your code. So you can say that the Operator pattern
is next level of automation which DevOps applies to Kubernetes. Of course, you could also go with Helm package manager 
and deploy/change your set of native resources via command like `helm update`, however this requires full-time engineer to sit and manage such operations by hands.

Operator workflow can shown via the following 3 steps:

{{ resize_image(path="k8s-operator/images/operator-pattern.png", width=800, height=600, op="fit_width") }}

1. User creates new custom resource instance via Kubernetes API, for example using `kubectl` CLI
2. Once customer resource is created and modified, this event is sent to custom controller
3. Once custom controller got the event it automatically creates to it by one or multiple CRUD commands towards API server

Later, users can modify or delete custom resources like any other native resource. Custom controller will handle these events as well and based
on its programmed logic decide what to do in case of custom resource modification or deletion. Custom controller can watch for native resources as well.

# Use Case: Kerberos Operator

Let's look at how to develop new operator from scratch in Scala using [Kubernetes-Client](https://github.com/joan38/kubernetes-client) library.

Our desired custom resource instance is going to be the following:

```yaml
apiVersion: io.github.novakov-alexey/v1
kind: Krb
metadata:
  name: my-krb
spec:
  realm: EXAMPLE.COM  
  principals:
    - name: client1
      password:
        type: static
        value: mypass
      keytab: cluster.keytab
      secret:
        type: Keytab
        name: cluster-keytab
    - name: user2
      keytab: cluster.keytab
      secret:
        type: KeytabAndPassword
        name: cluster-keytab
```

Based on resource, our operator will create/modify/delete the following native resources:
- Service to expose Kerberos server
- Pod to launch Kerberos container
- Secrets to keep Kerberos principals keytabs or credentials

Operator will have native resource templates in its code to apply them when it gets custom resource instance. Such native template will be 
used to set user variable taken from the custom resource and native template are going to be applied to Kubernetes. So that we will have logical
mapping between particular custom resource instance and its set of native resource which user expects to create/modify/delete in Kubernetes namespace(s).

{{ resize_image(path="k8s-operator/images/kerberos-workflow.png", width=700, height=600, op="fit_width") }}

## Implementation


### Libraries 

Add dependency to your Scala project, I am using SBT:

```scala
lazy val kubernetesClient = "com.goyeau" %% "kubernetes-client" % "0.8.0"
lazy val circeVersion = "0.14.1"
lazy val circeExtra = "io.circe" %% "circe-generic-extras" % circeVersion
lazy val circeCore = "io.circe" %% "circe-core" % circeVersion
```

### Resource Model

Our model custom resource specification will be encoded as hierarchy of case classes:

```scala
sealed trait Password
final case class Static(value: String) extends Password
case object Random extends Password

sealed trait Secret {
  val name: String
}

final case class Keytab(name: String) extends Secret
final case class KeytabAndPassword(name: String) extends Secret

final case class Principal(
    name: String,
    password: Password = Random,
    keytab: String,
    secret: Secret
)
final case class Principals(list: List[Principal])
final case class PrincipalsStatus(
    processed: Boolean,
    lastPrincipalCount: Int,
    totalPrincipalCount: Int,
    error: String = ""
)

final case class KrbServer(realm: String)
final case class KrbServerStatus(processed: Boolean, error: String = "")
```

In the result we will two top level custom resources: `Princpals` and `KrbServer`. Also, each custom resource has own status property `PrincipalsStatus`
and `KrbServerStatus` accordingly. Status property of the custom resource is additional way to communicate from your operator to user who is going 
to submit CR instances and changes to it. Status property schema is up to operator design, thus you can any structure in it.
Both status and main CR specification are validated according to in advance provided OpenAPI JSON schema. Basically, you provide JSON schema when you install
an operator to some of the Kubernetes namespace. Some operators can install CRD automatically, but they would require special RBAC permissions in their service account.

### Operator Main Class

First of all we need such classes to be imported in the Main class:

```scala
import cats.effect._
import cats.implicits._
import com.goyeau.kubernetes.client.EventType.{ADDED, DELETED, ERROR, MODIFIED}
import com.goyeau.kubernetes.client._
import com.goyeau.kubernetes.client.crd.{CrdContext, CustomResource}
import fs2.{Pipe, Stream}
import io.circe.generic.extras.auto._
import io.circe.{Decoder, Encoder}
import io.k8s.apiextensionsapiserver.pkg.apis.apiextensions.v1.CustomResourceDefinition
import krboperator.Controller.NewStatus
import krboperator.service.Template.implicits._
import krboperator.service.{Kadmin, Secrets, Template}
import org.typelevel.log4cats.Logger
import org.typelevel.log4cats.slf4j.Slf4jLogger

import java.io.File
import scala.concurrent.duration.{DurationInt, FiniteDuration}
import scala.reflect.ClassTag
```

Yes, there are a lot of them. This how much logic I put into one file. 

In the operator main class we will create Kubernetes client instance and later start watchers for `KrbServer` and `Principalas` custom resources.

```scala
object KrbOperator extends IOApp with Codecs {
  implicit def unsafeLogger[F[_]: Sync]: Logger[F] = Slf4jLogger.getLogger[F]

  val kubernetesClient =
    KubernetesClient[IO](
      KubeConfig.fromFile[IO](
        new File(s"${System.getProperty("user.home")}/.kube/config")
      )
    )

  override def run(args: List[String]): IO[ExitCode] = ???
  ....
```

### Watching Resource

Kubernetes-Client library is using FS2 library as direct dependency to watch for Kubernetes resources. That gives a lot of power
to handle events in fault-tolerant way. Also, FS2 nicely composes with Cats-Effect.

```scala
def watchCr[F[_]: Sync, 
    Resource: Encoder: Decoder: ClassTag, 
    Status: Encoder: Decoder](
      ctx: CrdContext
  )(implicit
      controller: Controller[F, Resource, Status],
      kubernetesClient: KubernetesClient[F]
  ) = {
    val context = crds.context[Resource]
    Stream.eval(
      Logger[F].info(s"Watching context: $context")
    ) >> kubernetesClient
      .customResources[Resource, Status](context)
      .watch()
      .through(handler[F, Resource, Status])
      .through(submitStatus[F, Resource, Status](ctx))
  }
```

Above code opens long-term HTTP connection to API-Server which is going to listen for custom resource updates. Until this connection is not closed by the
operator or by the API-Server, operator will pass custom resource event containing current resource state to function called `handler` and its output
will be passed to `submitStatus` function. These two functions will be called on every event sent by API-Server to us. Now, our goal is to 
react to this event and make something to happen in Kubernetes namespace(s). In our case, we need to create Kerberos pod, service to it and a bunch of secrets.
Also, our `Principals` resource means that principal list must be registered in Kerberos database using one of the Kerberos process called KDC. If you never worked with
Kerberos, then think of it like a database which we need to deploy to Kubernetes and create new users with requested credentials. Later, someone
will mount newly created secrets with keytabs to some user's Pod and will going to authenticate to Kerberos via keytab file. Keytab file is something like
password saved into a binary file with additional information.

### Handler Function

We are coming closer to processing the custom resource events. Below is a FS2 Pipe to take CustomResource type and return optional status
which is going to be added to the copy of original custom resource event. Then we take this optional status and to the next function as per our FS2 stream.

```scala
  def handler[F[_]: Sync, Resource, Status](implicit
      controller: Controller[F, Resource, Status]
  ): Pipe[F, Either[String, WatchEvent[
    CustomResource[Resource, Status]
  ]], Option[CustomResource[Resource, Status]]] =
    _.flatMap { s =>
      val action = s match {
        case Left(err) =>
          Logger[F].error(s"Error on watching events: $err") *> Sync[F].pure(
            none
          )
        case Right(event) =>
          Logger[F].debug(s"Received event: $event") >> {
            val status =
              event.`type` match {
                case ADDED =>
                  controller
                    .onAdd(event.`object`)
                    .map(addStatus(_, event.`object`))
                case MODIFIED =>
                  controller
                    .onModify(event.`object`)
                    .map(addStatus(_, event.`object`))
                case DELETED =>
                  controller.onDelete(event.`object`).map(_ => none)
                case ERROR =>
                  Logger[F].error(
                    s"Received error event for ${event.`object`}"
                  ) *> Sync[F].pure(none)
              }

            status.handleErrorWith { e =>
              Logger[F].error(e)(
                s"Controller failed to handle event type ${event.`type`}"
              ) *> Sync[F].pure(none)
            }
          }
      }
      Stream.eval(action)
    }
```

In the match statement, we call different methods of some controller object which will define right now:


```scala
import cats.effect.Sync
import cats.syntax.all._
import com.goyeau.kubernetes.client.crd.CustomResource
import krboperator.Controller.{NewStatus, NoStatus, noStatus}

import scala.language.implicitConversions

object Controller {
  type NewStatus[U] = Option[U]
  type NoStatus = NewStatus[Unit]

  def noStatus[F[_], U](implicit F: Sync[F]): F[Option[U]] = F.pure(none[U])
}

abstract class Controller[F[_], T, U](implicit val F: Sync[F]) {

  implicit def unitToNoStatus(unit: F[Unit]): F[NoStatus] =
    unit *> noStatus

  def onAdd(resource: CustomResource[T, U]): F[NewStatus[U]] = noStatus

  def onModify(resource: CustomResource[T, U]): F[NewStatus[U]] = noStatus

  def reconcile(resource: CustomResource[T, U]): F[NewStatus[U]] = noStatus

  def onError(resource: CustomResource[T, U]): F[NewStatus[U]] = noStatus

  def onDelete(resource: CustomResource[T, U]): F[Unit] = F.unit
}
```

As you might noticed, main class has word `Operator` in it. Event handlers are called controllers.
Basically, our entire application will be called operator. However, this operator app has internal logic wrapped into `Controller` class. 
We will have two controllers, each handles own custom resource such as `KrbServer` and `Principals`.

### Submit Status

Post handler function is to add status value and submit it back to API-Server:

```scala

  def submitStatus[F[_], 
      Resource: Encoder: Decoder, 
      Status: Encoder: Decoder](
      ctx: CrdContext
  )(implicit
      client: KubernetesClient[F],
      F: Sync[F]
  ): Pipe[F, Option[CustomResource[Resource, Status]], Unit] =
    _.flatMap { cr =>
      val action = cr match {
        case Some(r) =>
          for {
            name <- F.fromOption(
              r.metadata.flatMap(_.name),
              new RuntimeException(s"Resource name is empty: $r")
            )
            namespace <- F.fromOption(
              r.metadata.flatMap(_.namespace),
              new RuntimeException(s"Resource namespace is empty: $r")
            )
            _ <- client
              .customResources[Resource, Status](ctx)
              .namespace(namespace)
              .updateStatus(name, r)
              .flatMap(status =>
                if (status.isSuccess) F.unit
                else
                  Logger[F].error(new RuntimeException(status.toString()))(
                    "Status updated failed on Kubernetes side"
                  )
              )
          } yield ()
        case None => F.unit
      }

      Stream.eval(action.handleErrorWith { e =>
        Logger[F].error(e)(
          s"Failed to submit status"
        ) *> F.pure(none)
      })
    }
```

### Watchers and Reconcillers

We need to create a pair of watchers and return them to Cats IOApp to launch the entire application.
We are almost ready to define `run` method of the IOApp:

```scala
  override def run(args: List[String]): IO[ExitCode] =
    kubernetesClient
      .use { implicit client =>
        for {
          operatorCfg <- IO.fromEither(
            AppConfig.load
              .leftMap(e => new RuntimeException(s"failed to load config: $e"))
          )
          serverCtx = crds.context[KrbServer]
          _ <- createCrdIfAbsent[IO, KrbServer](
            serverCtx,
            crds.server.definition(serverCtx)
          )
          principalCtx = crds.context[Principals]
          _ <- createCrdIfAbsent[IO, Principals](
            principalCtx,
            crds.principal.definition(principalCtx)
          )
          _ <- createWatchers(
            operatorCfg,
            serverCtx,
            principalCtx
          ).parJoinUnbounded.compile.drain
        } yield ()
      }
      .as(ExitCode.Success)
  
  def createWatchers(
      operatorCfg: KrbOperatorCfg,
      serverCtx: CrdContext,
      principalCtx: CrdContext
  )(implicit client: KubernetesClient[IO]) = {
    val secrets = new Secrets(client, operatorCfg)
    val kadmin = new Kadmin(client, operatorCfg)
    val template = new Template(client, secrets, operatorCfg)

    implicit lazy val serverController =
      new ServerController(
        template,
        secrets,
        client,
        crds.context[KrbServer]
      )
    implicit lazy val principalController =
      new PrincipalController(
        secrets,
        kadmin,
        client,
        serverController,
        operatorCfg
      )

    Stream(
      watchCr[IO, KrbServer, KrbServerStatus](serverCtx),
      watchCr[IO, Principals, PrincipalsStatus](principalCtx),
      reconcile[IO, KrbServer, KrbServerStatus](serverCtx),
      reconcile[IO, Principals, PrincipalsStatus](principalCtx)
    )
  }
```

Before starting our watchers we check whether our CRDs are already registered in API-Server. If they are not, then make an attempt to register
by our operator:

```scala
  def createCrdIfAbsent[F[_]: Sync, Resource](
      ctx: CrdContext,
      customDefinition: CustomResourceDefinition,
      attempts: Int = 1
  )(implicit client: KubernetesClient[F]): F[Unit] = {
    val crdName = crds.crdName(ctx)
    for {
      found <- client.customResourceDefinitions
        .get(crdName)
        .map(_ => true)
        .recoverWith { case _ =>
          false.pure[F]
        }
      _ <-
        if (!found && attempts > 0)
          client.customResourceDefinitions.create(
            customDefinition
          ) *> createCrdIfAbsent(ctx, customDefinition, attempts - 1)
        else if (!found)
          Sync[F].raiseError(new RuntimeException(s"CRD '$crdName' not found"))
        else Sync[F].unit
    } yield ()
  }
```

If operator failed to create a CRD, then it fails the entire application, because it won't be able open watcher connection later for non-existing CRD.

Finally, let's look what `reconcile` function is doing:

```scala
  def reconcile[F[_]: Async, 
      Resource: Encoder: Decoder: ClassTag, 
      Status: Encoder: Decoder](
      ctx: CrdContext,
      every: FiniteDuration = 30.seconds
  )(implicit
      client: KubernetesClient[F],
      controller: Controller[F, Resource, Status]
  ): Stream[F, Unit] =
    Stream
      .awakeEvery(every)
      .evalMap { _ =>
        Logger[F].debug(
          s"Reconciling events for ${implicitly[ClassTag[Resource]].runtimeClass.getSimpleName}"
        ) *> client.customResources[Resource, Status](ctx).list()
      }
      .through(
        _.evalMap { crList =>
          Logger[F].debug(
            s"Got ${crList.items.length} custom resources"
          ) *> crList.items
            .map(cr => controller.reconcile(cr).map(addStatus(_, cr)))
            .sequence
        }.flatMap(Stream.emits)
      )
      .through(submitStatus[F, Resource, Status](ctx))
```

It calls every 30 seconds the API-Server to list all existing CR instances of a specific custom resource type. Then every instance in the list is checked
whether its state is correctly handled according to business logic of the operator. 
Main idea of reconciliation loop is to have a chance to recover from failures, which may occur in the main handler stream.

### Finish

At this point we have all pieces which can be used to implement any operator. Just plug in your own controller's logic.
In our case here, the main business logic is hidden in these classes:

```scala
val secrets = new Secrets(client, operatorCfg)
val kadmin = new Kadmin(client, operatorCfg)
val template = new Template(client, secrets, operatorCfg)
```

Full source of working operator can be found here: [https://github.com/novakov-alexey/krb-operator2](https://github.com/novakov-alexey/krb-operator2)

# Summary 

We have created an operator skeleton in Scala using Kubernetes-Client, which can be used to build any other operator.
Thanks to Cats-Effect and FS2 we can handle custom resource event stream easily composing functions. 

Kubernetes Operator is a dedicated application which automates deployment of complex stateful and stateless applications.
Such application can be implemented in any language since API-Server is exposing custom resource via REST API using JSON payloads.
There is no need to have special operator framework or library to enable developers to create new operators easily. Just keep a show case app
like we have developed and re-use its code to start another operator when needed. Basically, what is actually needed is a decent Kubernetes client
library like the one we used and you are good to go.
