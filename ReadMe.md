>_author_   :   oukohou  
>_time_     :   2019-09-26 16:44:48  
>_email_    :   oukohou@outlook.com

# A pytorch reimplementation of [SSR-Net](https://www.ijcai.org/proceedings/2018/0150.pdf).  
the official keras version is here: [SSR-Net](https://github.com/shamangary/SSR-Net)  

## results on [MegaAge_Asian](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/) datasets:  
|-|train|valid|test|
|:---:|:---:|:---:|:---:
|version_v1^[1]|train Loss: 22.0870 CA_3: 0.5108, CA_5: 0.7329|val Loss: 44.7439 CA_3: 0.4268, CA_5: 0.6225|test Loss: 35.6759 CA_3: 0.4935, CA_5: 0.6902
|original paper|**|**|CA_3: 0.549, CA_5: 0.741|
|version_v2^[2]|train Loss: 2.9401 CA_3: 0.6326, CA_5: 0.8123|val Loss: 4.7221 CA_3: 0.4438, CA_5: 0.6295|test Loss: 3.9311 CA_3: 0.5151, CA_5: 0.7163

[^1]: train from scratch, use MSEloss;  
[^2]: use pretrianed my implementation_v1, use L1Loss.


### Note:  
- This SSR-Net model can't fit big learning rate, learning rate should be smaller than 0.002.
otherwise the model will very likely always output 0, me myself suspects this is because of the 
utilizing Tanh as activation function.  
- And also: Batchsize [could severely affect the results](https://github.com/shamangary/SSR-Net/issues/38). A set of tested params can be :
    ```text
    batch_size = 50
    input_size = 64
    num_epochs = 90
    learning_rate = 0.001 # originally 0.001
    weight_decay = 1e-4 # originally 1e-4
    augment = False
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.L1Loss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    ```   
- The dataset preprocess is quite easy. For MegaAsian datasets, you can use the `./datasets/read_megaasina_data.py` directly;
for other datasets, just generate a pandas csv file in format like:
    ```text
    filename,age
    1.jpg,23
    ...
    ```
is OK. But also, remember to change the `./datasets/read_imdb_data.py` accordingly.
<br>

### onnxruntime C++ implementation  
thanks to [DefTruth](https://github.com/DefTruth) 's implementation here: [How to convert SSRNet to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_ssrnet-cn.md).    

#### another small note:
my reading understanding of [SSRNet]((https://www.ijcai.org/proceedings/2018/0150.pdf)) can be found:
 - on my [blog site](https://www.oukohou.wang/) here:[论文阅读-年龄估计_SSRNet](https://www.oukohou.wang/2019/09/20/SSRNet/) 
 - or on [zhihu](https://www.zhihu.com/) here: [论文阅读-年龄估计_SSRNet](https://zhuanlan.zhihu.com/p/87692466).  

which was written in Chinese. 