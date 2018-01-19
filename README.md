# Simpler&Easier_TF-SLIM_WORKFLOW(最简单的TF_SLIM无脑使用流程)

TF_SLIM主要用于图像分类的迁移学习（transfer learning），官网上的TF-SLIM库里有很多与新数据处理无关的脚本，当用自己的数据进行训练时也要对原有脚本做很多调整，所以我在官方TF_SLIM库的基础上删除很多跟新数据训练无关的脚本，并对剩下代码进行了修改，形成了一套通用迁移学习脚本库，你甚至不需要修改任何参数，只要按照我的要求把新数据图片及想要用于迁移学习的model放到指定文件夹里，即可无脑开始高大上的迁移学习！

对！你没有听错，你需要做的就只有那么多！就问你开不开心、刺不刺激、惊不惊喜蛤蛤蛤！

我们来过一下这个异常简单的流程：

1.将数据转化为TFRecord的形式，点击下图中MyProject文件夹：

![ABC](https://github.com/JackieGoGoGo/Simple-Optimized-TF_SLIM_WORKFLOW/blob/master/Readme_Pics/easier_TF-SLIM_workflow.png)


2.这个时候需要你将需要训练的图片放入到photos_to_be_converted文件夹中：

![ABC](https://github.com/JackieGoGoGo/Simple-Optimized-TF_SLIM_WORKFLOW/blob/master/Readme_Pics/photos_to_be_converted.png)


3.图片长宽大小无要求，但是需要你把相同标签的图片分到同一个文件夹中，文件夹就是其标签名：

![ABC](https://github.com/JackieGoGoGo/Simple-Optimized-TF_SLIM_WORKFLOW/blob/master/Readme_Pics/the_content_of_photos_folder.png)


4.将你需要迁移学习使用的模型的ckpt文件放入到transfer_model文件夹中：

![ABC](https://github.com/JackieGoGoGo/Simple-Optimized-TF_SLIM_WORKFLOW/blob/master/Readme_Pics/the_content_of_transfer_model_folder.png)


5.打开自己的终端，执行photos_convert_TFRcord脚本：python photos_convert_TFRecord ， 可以看到在MyProject中已经生成了相关TFRecord文件。

![ABC](https://github.com/JackieGoGoGo/Simple-Optimized-TF_SLIM_WORKFLOW/blob/master/Readme_Pics/converted.png)

6.我们可以开始执行train_image_classifier.py训练啦！！！执行下方的指令即可开始训练，当然如果你需要精细化地调参训练，可以详细看train_image_classifier.py中能够训练的参数。

python train_image_classifier.py --dataset_name=myproject --dataset_dir=./MyProject --train_dir=MyProject/train_logs/inception_v3 --dataset_split_name=train --model_name=inception_v3 --checkpoint_path=./MyProject/transfer_model/inception_v3.ckpt --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --clone_on_cpu=True

![ABC](https://github.com/JackieGoGoGo/Simple-Optimized-TF_SLIM_WORKFLOW/blob/master/Readme_Pics/training.png)

至此，所有你听说过的高大上的Nasnet、Inception-Resnet、Inception、VGG、ResNet，都能为你所用，让你们轻轻松松无脑获得迁移分类任务90%以上的准确率。
写得有点累了，关于预测相关的下次再更新哈！

