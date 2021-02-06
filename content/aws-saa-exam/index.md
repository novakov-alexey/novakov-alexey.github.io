+++
title="Path to AWS Solution Architect Certification"
date=2020-12-08
draft = false

[extra]
category="blog"
toc = true

[taxonomies]
tags = ["aws", "learning"]
categories = ["cloud"]
+++

Earlier this year I started learning few Udemy courses for [AWS Solution Architect Associate Certification](https://aws.amazon.com/certification/certified-solutions-architect-associate/). My initial plan was just to get some basic knowledge about AWS services, not the certification itself. Before starting the course, I did not have any experience with native AWS services. I never used EC2 instances to launch something in production. However, I used EC2 fleet of instances to deploy Kubernetes cluster. I started my path to cloud-native tools with Docker and shifted later to Kubernetes and OpenShift. Pure AWS services were never on my radar. I think the experience of deploying Kubernetes on top of the EC2 instances is one thing, but using different AWS services like ELB, EBS, EFS, EC2 and S3 is completely another story. So I decided it is time to get to know what are those AWS main services. After coming to the end of the Udemy course, I realised it worth to go for certification to make a final point in my AWS education. As I already spent so much time. 

{{ resize_image(path="aws-saa-exam/aws-exam.png", width=600, height=600, op="fit") }}
<!-- more -->
Eventually I got certification after 7 __calendar__ months. During that time, I did several pauses for vacation and other activities at work, when
learning AWS was too hard to keep in my head. Perhaps, pure learning time was from 3 to 4 months.

# Preparation

The final path I followed looks likes this:

1. Main Udemy preparation courses:
    - AWS Certified Solutions Architect Associate - 2020 SAA-C02 from Neal Davis
    - Ultimate AWS Certified Solutions Architect Associate 2020 from Stephane Maarek
1. AWS Free Video Courses
1. AWS related project at work ;-) 
1. Udemy SAA Practise Exams from Neal Davis
1. Udemy AWS VPC and Networking in Depth from Chetan Agrawal

The best learning was in a project at work. However, theoretical knowledge helped quiet a lot to be fast in practise. 

# How to pass the exam

Every Udemy preparation course contains a series of labs almost for every lecture. Let's say a block on AWS VPC (Virtual Public Cloud) has 15 lectures, then there are around 10 labs that you need to practise on your own.

Composition of the theoretical and practical labs at platform like Udemy plus real experience at work
gives you a high chance to pass the certification for "Associate" level. Recommended practical experience from Amazon to pass "Associate" level is 1 year.
I spent several times less with regards to practical experience. I believe those who already has practical experience of 1 year with AWS can easily
watch one of the MOOC course and straight away go for certification exam. If you want to go for certification in relaxed mode, then take your time
by working on some project at work related to AWS.

The hardest thing is to motivate yourself to go for certification after one or more years of working with AWS. I think some people will eventually loose the 
motivation in certification, since they are already so professional that they might think the certification is just waist of time. 
By end the day, the certification exam is also a stressful exercise. You either have to go to a certification centre in your city or go for online exam at home or office. So it gives you some headache.

# Recommendations

## Take notes

After watching and practising with topic number 10 out of 25 topics I quickly realised that I should have been taking some summary notes for
lecture I watched and practised before. So that I could quickly review my notes and recall those topics I learnt earlier. I started making an AWS mind map.
Since I was using mind maps for almost 10 years already, it was natural choice for me. The mind mapping software I use called [XMind](https://www.xmind.net/). 

So I started to do mind map in the middle of the first course, then I fulfilled the missing topics while watching second course, so that it covers
all possible topics related to Solution Architect Associate certification. Eventually, I got around 1600 nodes in my mind map tree. XMind software
performance is not good on my old 8 GB RAM laptop. It is probably not good practise to create such big mind maps. I could probably go with a separate mind map
per AWS topic or separate tab in the same map, but then I would ended up with 20 different maps or tabs. 
A text search across different mind maps would be a nightmare then. If you know a solution for big mind map problem, please let me know.

Below picture show some part of the AWS mind map I created as summary notes:

{{ resize_image(path="aws-saa-exam/mind-map.png", width=600, height=600, op="fit") }}

My mind map became so big that I can't really print it easily, if I would want that.

You can go with your own way of making notes. The important thing is to do them, rather than trying to keep everything in your head. Some people say
that hand-written notes are the best way to remember a lot in your head. I am always struggling with handwritten notes to transfer them into digital format.
There are some tools for that, but it is not straightforward. So I abandoned an idea of paper or tablet handwritten notes until some time in future.

One thing with regards to mind map, I think a mind map is useful only for the mind map author. If I read someone else's mind map, I can't remember it nor learn something with it. A mind map is a personal tool like personal notes in a notebook. There is only you who can understand your notes, since it reflects your brain work.

## Youtube

Do not waste time by watching free videos on youtube for AWS certification, since it is not going to be a complete course. They will probably not provide
practise test after each topic nor test exams with 65 questions each. This is the state of 2020 year. Maybe in future the things get change on Youtube, 
so that we will be able to do full blown MOOCs there as well. We will see.

# Exam Topics

I am software developer by trade. Back in my university I had no any strong focus on computer networking. This topic was the most difficult for me at AWS
certification. All those Sys Admin related things like block storage, VPN, firewall (security groups, network ACL), network file systems, encryption, DNS
took me several iteration until I finally learnt that. 

Below is a list of topics which are required from you to learn to pass the certification exam.

## 1. VPC

It is huge block that includes many things to get familiar with:
- IP ranges / CIDRs
- Gateway Endpoint
- VPC endpoint
- Security group, Network ACL
- Subnet
- NAT Gateway, Instance
- Site-to-Site VPN
- Direct Connect
- Transit Gateway
- Route table
- VPC flow logs
- Private Link
- VPC peering
- VPN CloudHub
- Interface endpoint

This topic was the most difficult for me. All that I listed above you need to know well before going to Solution Architect certification.

## 2. EC2 

It is also quite large and popular topic on the exam. Requires to know:
- Pricing Models: Reserved, On-Demand, Dedicated, Scheduled, etc.
- Relation to AMIs (VM images), location of AMIs, etc.
- Placement groups: cluster, spread, partition
- IP address types: private public, Elastic IP
- HPC features: EFA, Enhanced Networking
- EC2 instance lifecycle
- EC2 API: user, meta data

This topic was tricky for me as well, after being working for several years with Kubernetes. I did not care what underlying infrastructure was used.
VMs or bare-metal servers. Anyway, I found EC2 topic as important as VPC. Basically, it is one of the building block of the AWS. You must know
this topic very well.

## 3. ELB, Route53, CDN

Elastic Load Balancing is popular topic too. 
- General idea is to know how different load balancers can be used with EC2 instances, Target Groups.
- Different target types: EC2, IP, Lambda function
- When to use ALB, NLB or CLB
- Auto Scaling group connection with ELB and Target Groups
- Relation to CloudWatch and EC2 metrics
- Different features like Global Accelerator, HA setup using Availability Zones, stick sessions. multi-VPC balancing
- WAF
- DNS record sets
- Route 53 DNS policies (latency, failover, geo-proximity, etc.)
- CloudFront

## 4. Security

- KMS: key types from management perspective, supported encryption algorithms
- S3 data encryption types at rest (quite popular topic)
- Encryption support per data storage and database service: EBS, EFS, RDS, etc.
- High-level knowledge on different services for security like Certification Manager, Shield, Artefact

## 5. Databases & Data Processing

- RDS master and replica encryption. There are several lifecycle transitions to remember like how to make encrypted database from unencrypted database.
- Aurora, Serverless, Global
- DynamoDB
- Kinesis, Firehose
- RedShift
- Athena
- Glue
- EMR
- Data Migration Service
- ElasticCache: difference between Redis and Memcached
- DataSync
- Neptune

RDS, DynamoDB are the most popular topics on the exam. However, you need to know main capabilities of other databases/tools.

## 6. Integration

- SQS
- SNS
- Amazon MQ
- Step Functions
- System Manager Parameter Store
- SWF
- AppSync

You need to know the difference between SQS versus SNS very well.

## 7. IAM

- Users, Groups, Roles, Policies
- Authentication types
- Organization and SCP
- Resource Access Manager

## 8. Storage

As I mention, it was quite hard topic for me, since you need to remember the IOPS per each disk type,
storage class durability, availability, minimum storage period, retrieval period, prices
There are many sub-options to keep in mind. It is quite large topic and as popular on the exam as VPC and EC2.

- EBS
- EFS
- S3
- Instance Store
- Storage Gateway
- FSx for Windows, Lustre

## 9. Dev Tools

- Lambda
- API Gateway
- SAM
- ECS, ECR, Fargate, EKS
- CodeCommit, CodeBuild, CodeDeploy, CodePipeline (just know what they are offering)

## 10. Operations

- OpsWorks
- Cloud Formation
- Server Migration Service
- Recovery strategies: Backup and Restore, Pilot light, Warm Standby, Multi Site

## 11. Monitoring, Auditing

- CloudWatch Logs, Alarms, Events
- CloudTrail
- TrustedAdvisor
- AWS Config 

Those service that I have not commented above you just need high-level understanding what they do and that is enough.

# When and How to go for Exam

Once you feel it is time to get a certification exam, make sure you have passed through any test / simulation exam at Udemy, anywhere you can find them.
Check sample questions at AWS training web-site. 

In my case, it was very helpful to study right answers for failed questions at Udemy test exams.
Basically, these simulation tests already show you a percentage result whether you can pass real exam and how close you are to 72% of right answers.
I think if you above 80% right answers you can go for exam.

These days everything is happening remotely as well as certification at AWS. You can schedule an exam by make an order at "aws.training / certmetrics" web-sites.
Chose exam provider and day for an exam. 

In my case, I could schedule an exam on next day I checked the available time slots, so there is
no long time waiting to schedule an exam.

If you feel that you are already overloaded with the theory while preparing for an exam, then it is a sign that you are ready to for the certification.
Of course, need to try all possible test exams and see if you get good percentage of right answers. Some people do not pass from the first time, but
most of the time pass from the second attempt. So do not bother if you did not pass first time, it is going to be a good rehearsal since you will be able
to see real questions and get familiar with level of complexity. As for me, it was very important to know how difficult those real exam questions.
Once I started the exam I realised that I can cope with the real questions. 

One last recommendation, do not stay too long on one question during the real exam. I would say 1.5 minute per question
should be enough. Some questions can be answered within 10 seconds. It highly depends on the question length. Udemy test exams have lengthy question and that is quite close to reality.

# Summary

Although it requires quite a lot of time and efforts to go for any certification, it worth to do just because you will really learn a lot.
I would not learn that amount of knowledge in case I would just watch the video course. Also, certification may help you
to win a good project if you are working in a company or for yourself as a contractor. 
