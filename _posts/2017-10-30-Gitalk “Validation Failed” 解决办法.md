---
layout:     post
title:      Gitalk “Validation Failed” 解决办法
subtitle:   博客插件
date:       2017-10-30
catalog: true
tags:
    - Blog

---


> Last updated on 2018-12-07...

### 错误原因

Gitalk使用 window.location.pathname 创建 Issue 的 Tag，而而Github在今年更新了关于Issue的字数限制（50字）。

博客名超过4个中文字就出现Validation Failed。

### 解决思路

使用一个MD5工具，将location.pathname长度缩短。

### 具体操作

首先下载md5.js [md5.js](https://github.com/blueimp/JavaScript-MD5/tree/master/js)

gitalk 配置的最后一行要有个id属性

```code
gitalk:
  enable: true    #是否开启Gitalk评论
  clientID: 60110e51e72b4e2f40a9   #生成的clientID
  clientSecret: 13edafa5bafe7a06eb293737d6607cf6b7198bc7  #同上
  repo: coladrill.github.io    #仓库名称
  owner: ColaDrill    #github用户名
  admin: ColaDrill
  distractionFreeMode: true #是否启用类似FB的阴影遮罩
  id: 'window.location.pathname'
```

post.html中gitalk设置改为id: md5(window.location.pathname)

```javascript
<script src="/js/md5.js"></script>
<script type="text/javascript">
    var gitalk = new Gitalk({
    clientID: '{{site.gitalk.clientID}}',
    clientSecret: '{{site.gitalk.clientSecret}}',
    repo: '{{site.gitalk.repo}}',
    owner: '{{site.gitalk.owner}}',
    admin: ['{{site.gitalk.admin}}'],
    distractionFreeMode: {{site.gitalk.distractionFreeMode}},
    id: md5(window.location.pathname),
    });
    gitalk.render('gitalk-container');
</script>
```

效果如下，我们可以看到一串md5加密后的字符串
![](/img/post/20171030/1.png)


### 参考链接

- [港科大Calpa](https://calpa.me/2018/03/10/gitalk-error-validation-failed-442-solution/)
