---
layout:     post
title:      金融级分布式架构 SOFAStack
subtitle:   Docker、Kubernetes、蚂蚁金服
date:       2019-06-21

catalog: true
tags:
    - Develop
---


> Last updated on 2019-6-27...

6 月 24 日，国内云原生领域最重要的会议即将来袭！`KubeCon` + `CloudNativeCon` + `Open Source Summit China 2019`将在上海召开，蚂蚁金服此次也会重度参与，由多名技术专家进行分享并组织 workshop。

本次大会上，蚂蚁金服将会重点分享`Kubernetes 集群的管理`、`深度学习任务在 Kubernetes 上的大规模部署和调优`、`互联网金融`、`安全容器`等前沿课题。

从 2016 年起，蚂蚁金服开始深度使用 [Kubernetes（Made by Google）](https://kubernetes.io/zh/)，并且开源了自己的[金融级云原生分布式解决方案 SOFAStack](https://github.com/sofastack)，作为最终用户案例被 Cloud Native Computing Foundation (CNCF) 官方推荐。

> [以上消息链接](https://mp.weixin.qq.com/s/fm6A4RnVZv6Ts44Z4A5nig)、[会后总结](https://mp.weixin.qq.com/s/9krLbCDg6EaAZjJvoMl0qA)

> 吐槽一下，阿里总体风格上是一个缺少工程师文化的强业务驱动型公司，公司内部很多框架是基于谷歌开源框架封装再封装。

### 背景知识

![](/img/post/20190621/1.jpg)

#### IaaS 基础设施服务

2006年AWS推出EC2（Elastic Compute Cloud），作为第一代`IaaS（Infrastructure as a Service）`，用户可以通过AWS快速的申请到计算资源，并在上面部署自己的互联网服务。

**IaaS从本质上讲是服务器租赁并提供基础设施外包服务。**就比如我们用的水和电一样，我们不会自己去引入自来水和发电，而是直接从自来水公司和电网公司购入，并根据实际使用付费。

EC2真正对IT的改变是硬件的虚拟化（更细粒度的虚拟化），而EC2给用户带来了以下五个好处：
- 降低劳动力成本：减少了企业本身雇佣IT人员的成本
- 降低风险：不用再像自己运维物理机那样，担心各种意外风险，EC2有主机损坏，再申请一个就好了。
- 降低基础设施成本：可以按小时、周、月或者年为周期租用EC2。
- 扩展性：不必过早的预期基础设施采购，因为通过云厂商可以很快的获取。
- 节约时间成本：快速的获取资源开展业务实验。

以上是AWS为代表的公有云IaaS，还有使用OpenStack构建的私有云也能够提供IaaS能力。

#### PaaS 平台服务

**PaaS（Platform as a Service）是构建在IaaS之上的一种平台服务，提供操作系统安装、监控和服务发现等功能，用户只需要部署自己的应用即可。**

最早的一代是Heroku。Heroko是商业的PaaS，还有一个开源的PaaS——Cloud Foundry，用户可以基于它来构建私有PaaS，如果同时使用`公有云`和`私有云`，如果能在两者之间构建一个统一的PaaS，那就是`混合云`了。

> 企业出于安全考虑会希望将数据存放在私有云上面，同时又希望能使用公有云的免费资源，所以就将公有云与私有云进行合理的混合运用，将敏感数据或是运行关键性的工作负载放在私有云上面，而一般的工作或是需要扩展的工作放在公有云上面，达到安全又省钱的目的。

在PaaS上最广泛使用的技术就要数`docker`了，因为使用容器可以很清晰的描述应用程序，并保证环境一致性。管理云上的容器，可以称为是`CaaS（Container as a Service）`，如GCE（Google Container Engine）。也可以基于Kubernetes、Mesos这类开源软件构件自己的CaaS，不论是直接在IaaS构建还是基于PaaS。

PaaS是对软件的一个更高的抽象层次，已经接触到应用程序的运行环境本身，可以由开发者自定义，而不必接触更底层的操作系统。

#### Docker

软件开发所需的环境配置一般比较麻烦，往往换一台机器，就要重来一次。但如果软件可以`带环境安装`呢？也就是说，安装的时候，把原始环境一模一样地复制过来。

于是就有了两种思路：

**1、虚拟机（virtual machine）**

在一种操作系统里面运行另一种操作系统，比如在 Windows 系统里面运行 Linux 系统。应用程序对此毫无感知，因为虚拟机看上去跟真实系统一模一样，而对于底层系统来说，虚拟机就是一个普通文件，不需要了就删掉，对其他部分毫无影响。
- 主要缺点：虚拟机会独占一部分内存和硬盘空间。即使虚拟机里面的应用程序，真正使用的内存只有 1MB，虚拟机依然需要几百 MB 的内存才能运行。

**2、Linux 容器（Linux Containers，LXC）**

Linux 容器不是模拟一个完整的操作系统，而是对进程进行隔离。或者说，在正常进程的外面套了一个保护层。
对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。
由于容器是进程级别的，相比虚拟机有很多优势。
- 优点：启动快、资源占用少、体积小

**Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。**它是`目前最流行`的 Linux 容器解决方案。

Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。程序在这个虚拟容器里运行，就好像在真实的物理机上运行一样。有了 Docker，就不用担心环境问题。

总体来说，Docker 的接口相当简单，用户可以方便地创建和使用容器，把自己的应用放入容器。容器还可以进行版本管理、复制、分享、修改，就像管理普通的代码一样。

主要用途，目前有三大类。
- **提供一次性的环境：**比如，本地测试他人的软件、持续集成的时候提供单元测试和构建的环境。
- **提供弹性的云服务：**因为 Docker 容器可以随开随关，很适合动态扩容和缩容。
- **组建微服务架构：**通过多个容器，一台机器可以跑多个服务，因此在本机就可以模拟出微服务架构。

### [Kubernetes](https://kubernetes.io/zh/)

> [Kubernetes](https://zh.wikipedia.org/zh-hans/Kubernetes) 由Joe Beda、Brendan Burns和Craig McLuckie创立，其他谷歌工程师（包括Brian Grant和Tim Hockin等）进行加盟创作，由谷歌在2014年首次对外宣布 。

Kubernetes（常简称为K8s）是用于自动部署、扩展和管理容器化（containerized）应用程序的开源系统。
其遵循`主从式`架构设计。Kubernetes的组件可以分为管理单个的 node 组件和控制平面部分的组件。
![](/img/post/20190621/2.png)

**Kubernetes的基本调度单元**称为`pod`。通过该种抽象类别可以把更高级别的抽象内容增加到容器化组件。
- 一个pod一般包含一个或多个容器，这样可以保证它们一直位于主机上，并且可以共享资源。
- Kubernetes中的每个pod都被分配一个`唯一的（在集群内的）IP地址`这样就可以允许应用程序使用同一端口，而避免了发生冲突的问题。
- Pod可以定义一个卷，例如本地磁盘目录或网络磁盘，并将其暴露在pod中的一个容器之中。
- pod可以通过Kubernetes API手动管理，也可以委托给控制器来实现自动管理。

**Kubernetes服务**本质是一组协同工作的pod，类同多层架构（Multitier architecture）应用中的一层。构成服务的pod组通过标签选择器来定义。
- Kubernetes通过给服务分配`静态IP地址和域名`来提供服务发现机制（Service discovery），并且以`轮循调度`（Round-robin DNS）的方式将流量负载均衡到能与选择器匹配的pod的IP地址的网络连接上（即使是故障导致pod从一台机器移动到另一台机器）。
- 默认情况下，服务任务会暴露在集群中（例如，多个后端pod可能被分组成一个服务，前端pod的请求在它们之间负载平衡）；除此以外，服务任务也可以暴露在集群外部（例如，从客户端访问前端pod）。

### [SOFAStack]((https://tech.antfin.com/sofa))

全称Scalable Open Financial Architecture Stack，是蚂蚁金服自主研发的金融级分布式架构，包含了构建金融级云原生架构所需的各个组件，是在金融场景里锤炼出来的最佳实践。
![](/img/post/20190621/10.png)

蚂蚁金服从 2007 年开始从集中式架构走向分布式架构。我们把过去十多年的技术演进过程中自主研发的一套金融级分布式架构沉淀成为 SOFAStack™（Scalable Open Financial Architecture Stack）。

从 2007 年到 2012 年，蚂蚁金服完成所有业务系统的`模块化、服务化改造`。通过 TCC 模式解决了服务化、数据拆分等带来的数据一致性的问题，通过注册中心解决了服务单点的问题。

在完成服务化改造后，随着服务集群的增大，系统的伸缩性遇到了瓶颈，另外为了满足金融级的属性，蚂蚁金服对系统可用性、数据一致性提出了更高的要求。

蚂蚁金服从 2013 年开始摸索出了一套`单元化`的思想，并基于此，推出了`同城双活`、`异地多活`、`弹性调度等能力`，保证业务不停机，数据不丢失。

再之后随着国内互联网金融的崛起、蚂蚁金服的国际化，蚂蚁金服也将自己的能力和技术开放出来，在金融云上以云产品的形式存在，开发者可以基于此快速搭建金融级能力的分布式系统，同时我们也将内部的一些实践开源出来。

从 2017 年开始，`云原生的理念`正在快速发展，面对云原生带来的机会和改变，蚂蚁金服的策略是积极拥抱云原生。因为云原生带来的思想和理念刚好可以用来解决蚂蚁金服内部遇到的一些场景和问题。

从 2018 年 4 月 19 日宣布开源至今，SOFAStack 目前已经开源了包括 `SOFABoot`、 `SOFARPC`、`SOFALookout`、`SOFATracer`、`SOFARegistry` 等在内的一系列微服务相关的项目，并投入分布式事务 Seata 进行重要贡献。
![](/img/post/20190621/11.png)

> [sofastack开源](https://github.com/sofastack)、[seata开源](https://github.com/seata)

#### 微服务体系

![](/img/post/20190621/12.png)
SOFAStack 包含了构建微服务体系的众多组件，包括`研发框架`、`RPC 框架`，`服务注册中心`，`分布式链路追踪`，`Metrics监控度量`、`分布式事务框架`、`服务治理平台`等，结合社区优秀的开源产品，可以快速搭建一套完善的微服务体系。

#### 云原生架构

随着业务规模和系统复杂性的增长，传统的微服务管理运维也变得越来越快，开发人员和运维人员面临诸多挑战。`Service Mesh` 和 `Serverless` 等新兴概念的出现可以解决此类问题。

SOFAStack 为满足大规模部署下的性能要求以及应对落地实践中的实际情况，也开源了类似 `SOFAMesh` 和 `SOFAMosn` 等项目。

### [SOFADashboard](https://github.com/sofastack/sofa-dashboard-client)

SOFADashboard client 用于向 SOFADashboard 服务端注册 IP、端口、健康检查状态等应用基本信息。
并非是直接通过 API 调用的方式将自身应用信息直接注册到 SOFADashboard 服务端 ，而是借助于 `Zookeeper` 来完成：
![](/img/post/20190621/4.png)

客户端向 Zookeeper 中如上图所示的节点中写入数据，每一个 ip:port 节点代表一个应用实例，应用本身信息将写入当前节点的 data 中。

### [Service Mesh (sofa-mesh)](https://github.com/sofastack/sofa-mesh)

> [参考链接](http://www.servicemesher.com/blog/migrating-from-classical-soa-to-service-mesh-in-ant-financial/)

#### 多语言问题

SOFA 其实包含了金融级分布式中间件，CICD 以及 PAAS 平台。SOFA中间件部分包含的内容包括 `SOFABoot 研发框架`、`SOFA微服务相关的框架`（RPC，服务注册中心，批处理框架，动态配置等等）、`消息中间件`、`分布式事务`和`分布式数据访问`等等中间件。

以上的这些中间件都是基于 `Java` 技术栈的，目前 SOFA 在蚂蚁金服内部大概超过 `90%` 的系统在使用，除了这些系统之外，还有剩下的 `10%` 的系统，采用 `NodeJS，C++，Python` 等等技术栈研发的。这剩下的 10% 的系统想要融入到 SOFA 的整个体系中，一种办法是用对应的语言再去写一遍 SOFA 中间件的各个部分对应的客户端。

事实上，之前我们正是这么干的，蚂蚁金服内部之前就有一套用 NodeJS 搞的 SOFA 各个组件的客户端，但是最近几年随着 AI 等领域的兴起，C++ 也在蚂蚁金服内部也在被应用到越来越多的地方，那么我们是否也要用 C++ 再来写一遍 SOFA 中间件的各个客户端？

如果我们继续采用这种方式去支持 C++ 的系统，首先会遇到成本上的问题，每个语言一套中间件的客户端，这些中间件的客户端就像一个个烟囱，都需要独立地去维护，去升级。另一方面，从稳定性上来讲，之前 Java 的客户端踩过的一些坑，可能其他的语言又得重新再踩一遍坑。

对于多语言的问题来说，`Service Mesh`其实就很好地解决了这部分的问题，通过 Service Mesh的方案，我们可以尽量把最多的功能从中间件的客户端中移到 Sidecar 中，这样就可以做到一次实现，就搞定掉所有语言，这个对于基础设施团队来说，在成本和稳定性上都是一个提升。
![](/img/post/20190621/5.jpg)

#### 架构的渐进式演进

另外的一个问题其实是所有在往云原生架构中转型的公司都会遇到的问题，云原生看起来非常美好，但是我们怎么`渐进式的演进`到云原生的架构下？特别是对于遗留系统，到底怎么做比较好。

当然，一种简单粗暴的方式就是直接用云原生的设施和架构重新写一套，但是这样，投入的成本就非常高，而且重写就意味着可能会引入 Bug，导致线上的稳定性的问题。那么有没有一种方式可以让这些遗留系统非常便捷地享受到云原生带来的好处呢？

Service Mesh 其实为我们指明了一个方向，通过 Service Mesh，我们为遗留系统安上一个 `Sidecar`，少量地修改遗留系统的配置甚至不用修改遗留系统的配置就可以让遗留系统享受到服务发现，限流熔断，故障注入等等能力。
![](/img/post/20190621/6.jpg)

#### 升级过程中的耦合问题

最后我们在蚂蚁金服的服务化的过程中遇到的问题是中间件升级的问题，蚂蚁金融从单体应用演进到`服务化的架构`，再演进到`单元化的架构`，再演进到`弹性架构`，其实伴随了大量中间件升级，每次升级，中间件不用说要出新的版本去提供新的能力，业务系统也需要升级依赖的中间件，这中间再出个 Bug，又得重新升级一遍，不光是中间件研发同学痛苦，应用的研发同学也非常痛苦。

我们从单体应用演进到了服务化的架构，从原来好几个团队维护同一个应用，到各个团队去维护各自领域的应用，团队之间通过接口去沟通，已经将各个业务团队之间做到了最大程度的解耦，但是对于基础设施团队来说，还是和每一个业务团队耦合在一起。

我们中间尝试过用各种方法去解决升级过程中的耦合问题，一种是通过我们自己研发的应用服务器 `CloudEngine` 来管理所有的基础类库，尽量地去减少给用户带来的升级成本，不用让用户一个个升级依赖，一次升级就可以。

但是随着蚂蚁的业务的不断发展，规模地不断扩大，团队的数量，业务的规模和我们交付的效率已经成为了主要的矛盾，所以我们期望以更高的效率去研发基础设施，而不希望基础设施的迭代受制于这个规模。

后来蚂蚁自己研发的数据库 `OceanBase` 也在用一个 Proxy 的方式来屏蔽掉 OceanBase 本身的集群负载，FailOver切换等方面的逻辑，而刚好 Service Mesh 的这种 Sidecar 的模式也是这样的一个思路，这让我们看到将基础设施的能力从应用中下移到 Sidecar 这件事情是一个业界的整体的趋势和方向。
![](/img/post/20190621/7.jpg)

通过这种方式应用和中间件的基础设施从此成了两个进程，我们可以针对中间件的基础设施进行单独的升级，而不用和应用的发布升级绑定在一起，这不仅解放了应用研发和基础设施团队，也让基础设施团队的交付能力变地更强，以前可能需要通过半年或者一年甚至更长时间的折腾，才能够将基础设施团队提供的新的能力铺到所有的业务系统中去，现在我们通过一个月的时间，就可以将新能力让所有的业务系统享受到。这也让基础设施团队的中台能力变得更强了。

这样我们就可以把我们还是把一些架构当中非常关键的支撑点以及一些逻辑下沉到 Sidecar上面去，因为整个蚂蚁金服的整体架构有非常多的逻辑和能力承载在这一套架构上面的。这些东西我们有一个最大的职责是要支撑它快速向前演进和灵活。

### [Serverless](https://jimmysong.io/posts/what-is-serverless/)

Serverless（`无服务器架构`）指的是由开发者实现的服务端逻辑运行在无状态的计算容器中，它由事件触发， 完全被第三方管理，其业务层面的状态则被开发者使用的数据库和存储资源所记录。

下图来自谷歌云平台官网，是对云计算的一个很好的分层概括，其中 serverless 就是构建在虚拟机和容器之上的一层，与应用本身的关系更加密切。
![](/img/post/20190621/8.jpg)

许多年前，我们开发的软件还是`C/S`（客户端/服务器）和`MVC`（模型-试图-控制器）的形式，再后来有了`SOA`，最近几年又出现了微服务架构，更新一点的有`Cloud Native`（云原生）应用，企业应用从`单体架构`，到`服务化`，再到更细粒度的`微服务化`，应用开发之初就是为了应对互联网的特有的`高并发、不间断`的特性，需要`很高的性能`和`可扩展性`。

人们对软件开发的追求孜孜不倦，希望力求在软件开发的复杂度和效率之间达到一个平衡。但可惜的是，NO SILVER BULLET！几十年前（1975年）Fred Brooks就在The Mythical Man-Month中就写到了这句话。那么Serverlss会是那颗银弹吗？

Serverless不如IaaS和PaaS那么好理解，因为它通常包含了两个领域BaaS（Backend as a Service）和FaaS（Function as a Service）。

两者都为我们的计算资源提供了弹性的保障，BaaS其实依然是服务外包，而FaaS使我们更加关注应用程序的逻辑，两者使我们不需要关注应用程序所在的服务器，但实际上服务器依然是客观存在的。

#### BaaS

BaaS（Backend as a Service）后端即服务，一般是一个个的API调用后端或别人已经实现好的程序逻辑，比如身份验证服务Auth0，这些BaaS通常会用来管理数据，还有很多公有云上提供的我们常用的开源软件的商用服务，比如亚马逊的RDS可以替代我们自己部署的MySQL，还有各种其它数据库和存储服务。

#### FaaS

![](/img/post/20190621/9.jpg)
FaaS（Functions as a Service）函数即服务，FaaS是无服务器计算的额一种形式，当前使用最广泛的是AWS的Lambada。

现在当大家讨论Serverless的时候首先想到的就是FaaS，有点甚嚣尘上了。FaaS本质上是一种事件驱动的由消息触发的服务，FaaS供应商一般会集成各种同步和异步的事件源，通过订阅这些事件源，可以突发或者定期的触发函数运行。
