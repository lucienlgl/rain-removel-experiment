# 论文与数据集说明

[toc]
 ---

## 1、论文模型、代码、数据集
|Algorithms|Projects/Code Link|Paper Link|Data Set|
|:---:|:---:|:---:|:---:|
|JORDER|[Projects Link][pj-JORDER]|[Joint Rain Detection and Removal from a Single Image][pa-JORDER]|At Projects Link|
|DNN|[Projects Link][pj-DNN]|[Removing rain from single images via a deep detail network][pa-DNN]|At Projects Link|
|DNA|[Projects Link][pj-DNA]|[Clearing the Skies: A Deep Network Architecture for Single-Image Rain Removal][pa-DNA]|At Projects Link|
|DIN-MDN|[Code Link][pj-DIN-MDN]|[Density-aware Single Image De-raining using a Multi-stream Dense Network][pa-DIN-MDN]|At Code Link|
|ID-CGAN|[Code Link][pj-IDCGAN]|[Image De-raining Using a Conditional Generative Adversarial Network][pa-IDCGAN]|At Code Link|

*注：ID-CGAN论文只使用了数据集，并未实验其模型*

[pj-JORDER]: http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
[pa-JORDER]: https://arxiv.org/abs/1609.07769
[pj-DNN]: https://xueyangfu.github.io/projects/cvpr2017.html
[pa-DNN]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf
[pj-DNA]: https://xueyangfu.github.io/projects/tip2017.html
[pa-DNA]:https://xueyangfu.github.io/paper/2017/tip/tip2017.pdf
[pj-DIN-MDN]:https://github.com/hezhangsprinter/DID-MDN
[pa-DIN-MDN]:https://arxiv.org/abs/1802.07412
[pj-IDCGAN]:https://github.com/hezhangsprinter/ID-CGAN
[pa-IDCGAN]:https://arxiv.org/abs/1701.05957

## 2、实验模型与数据集说明

### 数据集说明
|Data Set|Train Data|Test Data|备注|
|:---:|:---:|:---:|:---|
|JORDER-dataset|Heavy-syn: 1800<br>Light-syn: 1800|Heavy-syn: 100<br>Light-syn: 200|-|
|DNN/DNA-dataset|syn: 900*14|syn: 100*14|*14表示将一张原始图片合成14张不同的雨图<br>（雨的纹理、大小等不同）|
|DIN-MDN-dataset|Heavy-syn: 4000<br>Medium-syn: 4000<br>Light-syn: 4000|test1-syn: 1200<br>test2-fu: 1000|1、heavy、medium、light的原始图片相同<br>2、test1-syn为作者合成的测试集，test2-fu为作者<br>从DNN-dataset中随机抽取组成的|
|ID-CGAN-dataset|syn: 700|syn: 100<br>nature: 52(无label)|-|
*syn：Synthetic Dataset*
>*部分论文还使用了Real-World Images，此处并未全部列举。因此类数据缺乏label，无法计算psnr和ssim，在此次实验中并未使用*

### 实验模型说明


## 3、实验结果

### DNN

|Data Set |PSNR average|SSIM average|备注|
|:---:|:---:|:---:|:---:|
|JORDER_dataset|29.4439|0.94165|-|
|ID-CGAN_dataset|24.0175|0.8509|-|
|	DIN-MDN_dataset|29.1188|0.90458|-|
|DNN/DNA-dataset|28.2303|0.92113|使用论文预训练模型与100*14测试集|
|论文中结果|-|0.90±0.05|计算了三张图片：女孩(0.90)、花(0.92)、雨伞(0.86)|

### DNA

|Data Set |PSNR average|SSIM average|备注|
|:---:|:---:|:---:|:---:|
|JORDER_dataset|22.3974|0.87608|-|
|ID-CGAN_dataset|22.2492|0.84489|-|
|	DIN-MDN_dataset|22.142|0.84398|-|
|DNN/DNA-dataset|24.061|0.88954|使用论文预训练模型与100*14测试集|


>**DNN&DNA 备注：**	
>* 在作者实验室主页下载的论文数据集划分与论文中描写有出入：论文中写的是9100训练集、4900测试集(0.65：0.35)，下载数据集为：12600训练集、1400数据集(0.9：0.1)
>* 论文中说明模型训练使用框架为caffe，而从作者实验室主页下载的训练代码为tensorflow，测试代码为matlab（测试代码只能运行获得去雨图片，具体的处理流程为p文件，加密无法查看）
>* 论文并没有根据heavy、medium、light拆分数据集训练，故训练时将JORDER数据集（heavy、light）和DIN-MDN数据集（heavy、medium、light）分别混合后，再进行训练、测试


### JORDER*

|Data Set |PSNR average|SSIM average|备注|
|:---:|:---:|:---:|:---:|
|ID-CGAN_dataset|L:18.9416<br>H:16.187|L:0.68443<br>H:0.60155|-|
|DIN-MDN_dataset|L:20.7581<br>H:17.7029|L:0.78747<br>H:0.71711|-|
|DNN/DNA-dataset|L:21.234<br>H:17.8687|L:0.82059<br>H:0.74315|-|
|JORDER_dataset|L:20.0298<br>H:17.1231|L:0.79724<br>H:0.71692|此处的H使用了H100测试集，L使用L200测试集|

>此处使用的是论文作者的预训练模型（论文5.2章节去雨+雨雾的模型），L表示使用的是去除小雨的参数（即heavy_case设为0），H表示使用的是去除大雨的参数（即heavy_case设为1）。由于其他未对其他数据集进行训练的原因：
>* 其他数据集缺乏雨水层S和雨水区域层R的数据；
>* 论文未给出训练代码





