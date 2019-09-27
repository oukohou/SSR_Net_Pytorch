>_author_   :   oukohou  
>_time_     :   2019-09-26 16:44:48  
>_email_    :   oukohou@outlook.com

# A pytorch reimplementation of [SSR-Net](https://www.ijcai.org/proceedings/2018/0150.pdf).  
the official keras version is here: [SSR-Net](https://github.com/shamangary/SSR-Net)  

## results on [MegaAge_Asian](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/) datasets:  
|-|train|valid|test|
|:---:|:---:|:---:|:---:
|my implementation|train Loss: 22.0870 CA_3: 0.5108, CA_5: 0.7329|val Loss: 44.7439 CA_3: 0.4268, CA_5: 0.6225|test Loss: 35.6759 CA_3: 0.4935, CA_5: 0.6902
|original paper|**|**|CA_3: 0.549, CA_5: 0.741|



### Note:  
- This SSR-Net model can't fit big learning rate, learning rate should be smaller than 0.002.
otherwise the model will very likely always output 0, me myself suspects this is because of the 
utilizing Tanh as activation function.  
- And also: Batchsize [could severely affect the results](https://github.com/shamangary/SSR-Net/issues/38).   

