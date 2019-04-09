# CaptchaRec
验证码识别，几种方案：KNN，CNN...

# 项目简介：
最近在做一个有趣的项目，需要对某网站的验证码进行识别。

某网站验证码如图：![](https://oscimg.oschina.net/oscnet/40a40aea5e1174da3e6747717106114d279.jpg)，像素大小：30x106x3

通过人工标记的验证码数量约为1000条：
![](https://oscimg.oschina.net/oscnet/f5775cd0203629b807976712716e5980176.jpg)

现在需要通过机器学习方法来进行识别新验证码，设计的方案有如下四种：

- KNN + 原样本图；需要对图像去噪、二值化、切割等处理。对数据量要求没CNN高。
- CNN + 原样本图；缺点：样本少，优点：数据质量高。
- CNN + 构造类似验证码图；缺点：构造验证码是否和和原验证码类似，需要较高技术；优点：样本多。

- CNN + 单字符样本图；优点：输入图像小，且输出类别少。

- 其他：如用pytesseract+去噪+二值化等，简单尝试了一下准确率很低，pass掉了


# 方案一：KNN + 原样本图
步骤：
 - 去噪：	 原图：![](https://oscimg.oschina.net/oscnet/12c0031ac74d8128eb7174270b91af94031.jpg)

	 去噪后：![](https://oscimg.oschina.net/oscnet/d06fa2e42b3e80207c007abc81cb62a7276.jpg)

- 切分最小图


	切分后：![](https://oscimg.oschina.net/oscnet/5fed49de7ece64d52929d2e9e000ca999ae.jpg)

- 分割字符串：

	分割后：![](https://oscimg.oschina.net/oscnet/c1f8c4323959a9e6358f79e15b8949c4c98.jpg)，![](https://oscimg.oschina.net/oscnet/e204d490ee3eea0a2a8b4a38f04848da8f6.jpg)，![](https://oscimg.oschina.net/oscnet/ddc016d4fc296e5f21b0e4967904c64659a.jpg)，![](https://oscimg.oschina.net/oscnet/adecb42f7c6f3bf429336148edf9d313ca4.jpg)
	

	
	经过对1000张标记好的图片进行处理，得到各个字母数字对应的单字符图片数据集：
	![](https://oscimg.oschina.net/oscnet/c2ff89a0618cf14ffe5bcd2a2d9586859ba.jpg)
	![](https://oscimg.oschina.net/oscnet/7488281a91e93f843b8498851df6136f819.jpg)
- KNN训练及预测：
	
	对图像进行灰度处理

- 运行结果：（单个字符预测精度），KNN最高，达到80%，而SVM,DT,LR均较低
	```
	KNN accuracy score: 0.8170731707317073
	SVM accuracy score: 0.6341463414634146
	DT accuracy score: 0.4878048780487805
	LR accuracy score: 0.5975609756097561
	```
	KNN 预测图片：![](https://oscimg.oschina.net/oscnet/c6f491ee9732a7d6b8bc34fe0750b8ff0d8.jpg)
	```
	mHFM
	crdN
	wa5Y
	swFn
	ApB9
	eBrN
	rJpH
	fd9e
	kTVt
	t7ng
	```





# 方案二：CNN+原样本图

步骤：
- 处理样本数据1020张图，灰度化 ，像素大小30*106，标签为小写字符（标记人员太懒了）；

- 拆分数据：train:80%, val:20%

- 网络模型：输入数据维度30*106,采用三层CNN，每一层输出特征维数分别：16,128,16，FC层输出 512维,最后全连接输出4x63，每行代表预测字符的概率。

- 结果：验证集字符准确率最高到达了50%

# 方案三： CNN+ 构造类似验证码图
第三方库生成的验证码如下所示：
```
from captcha.image import ImageCaptcha  # pip install captcha
```
![](https://oscimg.oschina.net/oscnet/1ff59f7d6bde8ddd1caa50dca820f6156bb.jpg)

下载相应的字体(比较难找)，然后修改第三方库中image.py文件，修改了第三方库后生成的验证码：
![](https://oscimg.oschina.net/oscnet/2638db8d2fe2e571d996a4a7fe5ccda8e9d.jpg)

![](https://oscimg.oschina.net/oscnet/1623001b334f34002e50125f0ef8cc6dc21.jpg)

效果和我们需要的验证码比较相似了，但还是有区别。
```
    fonts = ["font/TruenoBdOlIt.otf", "font/Euro Bold.ttf", "STCAIYUN.TTF"]
    image = ImageCaptcha(width=106, height=30,fonts=[fonts[0]],font_sizes=(18,18,18))
    captcha = image.generate(captcha_text)
```
image.py
```
略..
```

采用自动生成的验证码，用于CNN训练模型，训练和验证精度都达到了98%，但测试原图1000样本的字符精度最高只有40%,由此可见，生成的验证码还是与目标验证码相差较大。
```
step: 18580/20000...  loss: 0.0105... 
step: 18600/20000...  loss: 0.0121... 
step: 18600/20000...  --------- val_acc: 0.9675     best: 0.9775  --------- test_acc2: 0.4032 
step: 18620/20000...  loss: 0.0131... 
step: 18640/20000...  loss: 0.0139... 
step: 18660/20000...  loss: 0.0135... 
step: 18680/20000...  loss: 0.0156... 
step: 18700/20000...  loss: 0.0109... 
step: 18700/20000...  --------- val_acc: 0.9625     best: 0.9775  --------- test_acc2: 0.3995 
```

# 方案四： CNN+ 字符样本集
由于只有1000样本，直接经过CNN端到端输出字符序列，很难到达精度要求，为此方案三采用自动创建样本集的方法，但样本质量和真实样本之间存在差异，导致预测不准。为此，将原1000样本进行分割处理为单字符集，样本数量约4000左右，且输入维度减小很多，同时输出类别也减小很多。分析后改方案有一定可行性。

样本集处理与之前KNN一样：
	经过对1000张标记好的图片进行处理，得到各个字母数字对应的单字符图片数据集：
	![](https://oscimg.oschina.net/oscnet/c2ff89a0618cf14ffe5bcd2a2d9586859ba.jpg)
	![](https://oscimg.oschina.net/oscnet/7488281a91e93f843b8498851df6136f819.jpg)
	


- 运行结果：字符预测精度95%以上
```
step: 2500/200000...  loss: 0.0803... 
step: 2500/200000...  --------- acc: 0.0854     best: 0.1341 
step: 2520/200000...  loss: 0.0818... 
step: 2540/200000...  loss: 0.0844... 
step: 2560/200000...  loss: 0.0827... 
step: 2580/200000...  loss: 0.0794... 
step: 2600/200000...  loss: 0.0823... 
step: 2600/200000...  --------- acc: 0.1951     best: 0.1341 
save best model...
step: 2620/200000...  loss: 0.0775... 
step: 2640/200000...  loss: 0.0754... 
step: 2660/200000...  loss: 0.0823... 
step: 2680/200000...  loss: 0.0678... 
step: 2700/200000...  loss: 0.0763... 
step: 2700/200000...  --------- acc: 0.3049     best: 0.1951 
.
.
.
.
step: 41400/200000...  --------- acc: 0.8659     best: 0.9512 
step: 41450/200000...  loss: 0.0091... 
step: 41500/200000...  loss: 0.0134... 
step: 41550/200000...  loss: 0.0151... 
step: 41600/200000...  loss: 0.0256... 
step: 41600/200000...  --------- acc: 0.9390     best: 0.9512 
```

预测图片：![](https://oscimg.oschina.net/oscnet/c6f491ee9732a7d6b8bc34fe0750b8ff0d8.jpg)
```
mHPM
srdN
wa5Y
eWpn
AgB9
eHr8
rJpH
fd9e
bTYt
tTwg
```

# 增加滑动验证码识别功能
 
 slip_captcha/
 
 ![](https://p3-dy.bytecdn.cn/img/security-captcha/slide_4b1eae0047860b6fb4428c728dad012616f17b39_1_1.jpg~tplv-obj.image)
 ![](https://p3-dy.bytecdn.cn/img/security-captcha/slide_4b1eae0047860b6fb4428c728dad012616f17b39_2_1.png~tplv-obj.image)
 
 
