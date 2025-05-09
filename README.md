# deepseek-on-vllm-VRAMandDRAM
一套基于Vllm的显存内存混合模式大模型部署工具（图形界面），VRAMandDRAM模式虽然慢一点，但是解决了超大模型在普通家用计算机上的部署问题。

# 使用协议-（非商业用途）支持学习与探索更多用途
本软件禁止以下用途，如果一定要使用，请先获取授权：
- 直接或间接的商业服务（如API收费、售卖模型权重、公有云）。
- 集成到商业产品中。
- 用于整体售卖的私有云部署服务。

# 作者和联系方式

作者：老谭
邮箱：10267672@qq.com
B站：https://space.bilibili.com/328484347

# 程序名称
vllm-gui-server-r1-loggood.py （这个版本时只用显存的模式，适合有多块大容量显存N卡的用户）
vllm-gui-server-r1-mem-good4.py （这个版本带mem的是支持内存显存混合模式）

# 本项目VLLM+VRAM+DRAM 开发心得

自从deepseek问世后，个人部署、私有部署deepseek满血版或者蒸馏版的需求呈爆炸性增长，虽然大家部署后也不知到用来干嘛（我是做行业应用的我是知道的），但很热情的拥抱AI，太好了。
但是大家的显卡也不是，大多是消费级显卡，显存非常有限，如何在有限的显存里玩出花样就成了一个课题。

VLLM是一个不错的平台，比Ollama强，速度快，但原生的Vllm不支持混合内存模型部署。

除了追加显卡，另外一个办法就是显存内存混合使用，也有人叫这是统一内存（Unified Memory Management），但实际上老黄对统一内存的硬件要求很高，一般的计算机达不到，想达到只能买新计算机最少DDR5内存以上。

我们这个项目是基于DDR3内存开发，首先保证能跑得动，再讨论跑得快的问题。

在此之前的内存显存混合模式跑大模型的软件有一些，各有特点。本软件的特点是：
1、先把大模型完全加载到内存，再由内存加载到显存，这样做就避免了先加载到显存可能出现的报错。
2、内存加载完毕后，运行时采用动态优化。
3、图形化界面，省去了繁杂的命令行操作。
4、支持跨平台 Windows、Ubuntu，都支持。
