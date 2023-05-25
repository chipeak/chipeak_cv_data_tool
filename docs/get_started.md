## 环境安装:
	首次安装：pip install ccdt
	版本更新安装：两种方式，（1）先卸载 pip uninstall ccdt，在安装即最新版本 pip install ccdt。（2）知道最新版本号，直接 pip install ccdt==2.1.0
	安装时主要依赖模块：tqdm、opencv_python、numpy、pycocotools、prettytable、shapely、shapely、Pillow、psutil、aiofiles、pypinyin
	注意：如果直接安装ccdt相关依赖出错时，手动单独安装依赖模块，并排查window安装环境针对c++依赖安装问题

## 工程安装包说明：
	芯峰科技，计算机视觉，数据处理工具包，chipeak computer vision data tool。简称ccdt
	参数说明：必填参数，input_dir：输入路径，output-dir：输出路径。其余参数根据具体功能选择填参。

## Linux使用指令
请参考[Linux入门文档](./linux_command_line_usage.md)学习基本使用。

## window使用指令
请参考[window入门文档](./window_command_line_usage.md)学习基本使用。

## pycharm开发使用指令如下所示

# 一、labelme数据集处理

## 1、指定份数抽取，拷贝，输出目录可以自定义，如果输入目录和输出目录相同，表示存放在输入目录下。
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=extract
--output-format=labelme
--extract-portion=3
--label-name=oil
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\extract
```

## 2、指定张数抽取，剪切或移动
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=extract
--output-format=labelme
--extract-amount=5
--select-cut
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\cut
```

## 3、指定张数抽取，拷贝
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=extract
--output-format=labelme
--extract-amount=5
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\chouqu
```

## 4、打印labelme数据集信息，包含图像张数、标注属性、未标注数量
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=print
```

## 5、重命名label标签名称，拷贝一份后对json文件中的内容进行修改
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=rename
--output-format=labelme
--rename-attribute={'label':{'car':'car_modify','plate':'plate_modify',}}
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\rename
```

## 6、重命名label标签名称，在当前输入路径下重写json，不用拷贝一份，把输入路径和输出路径写一样的即可
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original66'}]"
--function=rename
--output-format=labelme
--rename-attribute={'label':{'car':'car_modify','plate':'plate_modify',}}
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original66
```

## 7、筛选正样本数据集
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=filter_positive
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_positive
```

## 8、筛选负样本数据集
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=filter_negative
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_negative
```

## 9、筛选标注label类别数据集，默认正样本，自动筛选，无需填写label标签
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=filter_label
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_label_automatic
```

## 10、筛选标注label类别数据集，默认正样本，填写label标签
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=filter_label
--filter-label="['car','plate']"
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_label_appoint
```

## 11、筛选标注label下的flags类别属性数据集，默认正样本，填写label标签下的属性flags：{'other': False, 'yellow_green': False, 'yellow': True, 'blue': False, 'green': False, 'white': False, 'clear': True, 'incomplete': False}
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=filter_flags
--filter-flags="['blue','green','yellow']"
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_flags_appoint
```

## 注意：同时指定label类别与flags类别属性进行筛选数据集，功能没有开发。

## 12、检查imagePath路径，是否符合..\\00.images\\*.jpg，把不符合要求的json文件重写。输入路径与输出路径保持一致即可。
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original8'}]"
--function=check_image_path
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original8
```

## 13、合并根据label筛选后的数据集，保障图像唯一不重复
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_label_automatic'}]"
--function=merge
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_merge
```

## 14、指定label标签名称删除标注的shape标注
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_merge'}]"
--function=delete
--output-format=labelme
--del-label="['car']"
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\delete_label
```

## 15、对labelme数据集进行去重，图像重复、json文件重复就删除，以文件MD5值作为唯一判断标准。异步读数据，同步删除数据，功能可优化。
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_label_automatic'}]"
--function=duplicate
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\filter_label_automatic
```

## 16、labelme转coco，包含标注矩形框转coco，多边形转coco，多边形+矩形框组合标注转coco，多边形+矩形框打组标注转coco
```
"[{'type':'Coco','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=convert
--output-format=coco
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\convert_coco
```

## 17、labelme转coco，包含标注矩形框转coco，多边形转coco，多边形+矩形框组合标注转coco，多边形+矩形框打组标注转coco，图片路径改变为存储对象访问路径，--http-url=http://192.168.1.235:9393/chipeak-dataset，http-url根据对象存储服务动态改变
```
"[{'type':'Coco','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=convert
--output-format=coco
--http-url=http://192.168.1.235:9393/chipeak-dataset
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\convert_coco
```

## 18、针对矩形框标注的车后，又标注了多边形车牌，建立分组关系。即把有车和车牌的标注分为一组，并用group_id建立关系关系。（也存在自动打组分错问题，需要人工检查）
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=relation
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\relation
```

## 19、根据矩形框（车）+多边形框（车牌）标注分组的数据进行抠图，并保存为labelme。被截取的目标为矩形框，被保存的画框目标为矩形框内包含的多边形框。异步读数据，同步写数据，功能可优化。
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=matting
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\matting
```

## 20、压缩指定份数抽取的labelme数据集。同步读写数据，功能可优化。
```
ccdt \
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\extract'}]"
--function=compress
--sync
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\compress
```

## 21、检查标注分组数量是否正确。常见错误包含：一组标注少一个标注框或点，一组标注的group_id值不对。通过分组数量进行判断。
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=check
--judging-group=5
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\check_group
```

## 22、检查标注flags的默认值（False）是否被修改为True，约定必须修改的没有修改就判断为错误。比如：检查车牌颜色、单双层，是否，漏勾选。但不能检查勾选错误。譬如：complete被勾选，但车牌号没有录入
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=check
--judging-flags={'plate':['yellow_green','yellow','blue','green','white','black','single','double','complete','incomplete']}
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\check_flags
```

## 23、检查车牌多边形不为4个点的情况，可以提前检查，
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=check
--judging-polygon=4
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\check
```

## 24、检查标注坐标位置，是否超越原始图像宽高边界
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=check
--judging-cross-the-border=polygon
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\check
```

## 25、截取多边形标注的车牌，并把倾斜的进行矫正，并把多边形点进行左上、右上、右下、左下的顺序排列，自动分出单层车牌与双层车牌
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=correct
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\correct
```

## 26、截取多边形标注的车牌，不做任何排序，不做畸变矫正，自动分出单层车牌与双层车牌
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original'}]"
--function=original
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\correct
```

## 27、针对标注形状超越图像边界情况，使用程序自动矫正
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\check'}]"
--function=cross
--automatic-correction
--output-format=labelme
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\check
```

## 28、根据labelme的text字段的文本内容抽取数据，比如根据阈值、车牌号
```
"[{'type':'BaseLabelme','input_dir':'H:\1.model_train\7.SDC\08.fall_fall_doubt-zy-20230406\result_images\huawei_yan_xuan_filter_positive\_laonianhuodongzhongxin1'}]"
--function=extract
--output-format=labelme
--extract-text="['0.7','0.8','0.9']"
--output-dir=H:\1.model_train\7.SDC\08.fall_fall_doubt-zy-20230406\result_images\text0
```

## 29、递归修改文件目录路径，把文件目录路径的汉字转拼音后，重命名文件目录路径。同步读写数据，功能可优化。
```
"[{'type':'BaseLabelme','input_dir':'H:\1.model_train\7.SDC\08.fall_fall_doubt-zy-20230406\result_images\huawei_yan_xuan_filter_positive\_老年活动中心1'}]"
--function=pinyin
--sync
--output-format=labelme
--output-dir=H:\1.model_train\7.SDC\08.fall_fall_doubt-zy-20230406\result_images\huawei_yan_xuan_filter_positive\_老年活动中心1
```

## 30、用模型跑出来的矩形框，替换标注的矩形框。先删除标注矩形框，然后合并。input_dir：填写原始人工标注的labelme数据集输入路径，model_input_dir：填写模型预测的labelme数据集输入路径，--del-label：填写原始人工标注的labelme数据的矩形框label标签
```
"[{'type':'BaseLabelme','input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\original','model_input_dir':'W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\om_test'}]"
--function=merge
--rectangle-merge
--output-format=labelme
--del-label="['car']"
--output-dir=W:\my_work\11.hsq\jiyi\HKplate\20230420\zy\merge
```

# 二、视频数据处理，支持视频格式有（.mp4, .MP4, .mov. .avi .MOV .264 .dav .wmv .AVI .avi .webm .mkv .WMV .FLV .flv .MPG .mpg）

## 1、提取视频帧功能，包含单个视频文件提取和多个视频文件提取
```
--input=D:\ccdt\datasets_video\1.wmv
--function=split
--interval=50
--filename-format="{:0>8d}.jpg"
--save-images=00.images
--save-labelme=01.labelme
--output-dir=D:\ccdt\datasets_video\video
```

## 2、多个视频文件提取，指令参考如下，--interval=50（50帧抽一张）
```
--input=D:\ccdt\datasets_video
--function=split
--interval=50
--filename-format="{:0>8d}.jpg"
--save-images=00.images
--save-labelme=01.labelme
--output-dir=D:\ccdt\datasets_video\video
```

# 三、coco数据集处理

## 1、coco数据集转labelme数据集，功能暂未开发