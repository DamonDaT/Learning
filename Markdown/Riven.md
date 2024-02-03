#### 【说明】

***

记录一些好玩的事情（因为个人比较喜欢LOL中的Riven，故以此命名该文件）



#### 【一】个人深度学习服务器

***

> 处于学习的目的，搭建了一台属于自己的深度学习服务器，第一次玩，记录一下配置和效果

* 机箱：联力包豪斯EVO全视版 白色
* 风扇：积木三代若干
* 主板：微星 MEG Z790 ACE 战神（EATX的板子，后续考虑扩展多一张卡，但很大，海景房差点放不下）
* 显卡：七彩虹 RTX4090 水神
* CPU：Intel 13代 19-13900KF
* 水冷：先马 SAMA XW360DW（带IPS屏）
* 内存： 金百达 KINGBANK DDR5 6400 32GB X2
* 固态：三星 MZ-V9P2T0BW 2TB PCIe4.0x4 
* 电源：海盗船 RM1200x SHIFT（选了SHIFT的导致走线很麻烦... 要跟机箱配合着来选）

* 系统：Ubuntu 22.04 LTS



#### 【二】FRP 内网穿透

***

> 处于学习的目的，搭建深度学习服务器，

* 版本：v0.54.0 https://github.com/fatedier/frp/releases

<img src="/home/dateng/code/Learning/Markdown/image/Riven/1.1.jpg">



##### 【2.0】准备工作

***

* 个人深度学习服务器（Ubuntu 22.04 LTS）
* 云服务器（选择了阿里云 ECS 实例也是 Ubuntu 22.04 LTS）



##### 【2.1】Frps (服务端)

***

* ssh登录云服务器实例（秘钥认证或密码认证 官网有介绍这里不多赘述）

https://help.aliyun.com/zh/ecs/user-guide/connect-to-an-instance-by-using-third-party-client-tools/?spm=a2c4g.11186623.0.0.229622d5htvyYb

* 在 /usr/local/bin 目录下创建 frp文件夹

```she
sudo mkdir /usr/local/bin/frp
```

* 解压下载好的 frp 压缩包

```shell
tar -xzvf frp_0.54.0_linux_amd64.tar.gz
```

* 将目录下的所有内容都拷贝到 /usr/local/bin/frp 目录下

```shell
cp frp_0.54.0_linux_amd64_xxx/* /usr/local/bin/frp
```

* 修改 frp 文件夹和文件权限

```shell
chmod 755 -R /usr/local/bin/frp
```

* 修改 frps.toml 文件内容

```shell
bindAddr = "0.0.0.0"
bindPort = 7000                     # frp 服务器的端口号
webServer.addr = "0.0.0.0"
webServer.port = 7500               # frp 网页侧的端口号
webServer.user = "xxx"              # frp 网页侧的登录用户名
webServer.password = "xxx"          # frp 网页侧的登录密码
auth.method = "token"               # frp 连接认证方式
auth.token = "xxx"                  # frp 连接口令
```

* 添加 system 脚本：创建 /etc/systemd/system/frps.service 文件，权限755，内容如下

```shell
[Unit]
Description=Frp Server Service
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=30s
ExecStart=/usr/local/bin/frp/frps -c /usr/local/bin/frp/frps.toml
ExecStop=/usr/bin/killall frps
killMode=control-group

[Install]
WantedBy=multi-user.target
```



