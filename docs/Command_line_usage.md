## 抠图，单数据集处理输入指令参考
```
ccdt
--input_datasets="[{'format': 'labelme','images_dir': 'Z:/4.my_work/9.zy/00/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/00/01.labelme'},]"
--input_dir=Z:/4.my_work/9.zy
--output_dir=Z:/4.my_work/9.zy/test
--function=matting
```

## 抠图，多数据集同时处理输入指令参考
```
ccdt
--input_datasets="[{'format': 'labelme','images_dir': 'Z:/4.my_work/9.zy/00/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/00/01.labelme'},{'format': 'labelme','images_dir': 'Z:/4.my_work/9.zy/11/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/11/01.labelme'},]"
--input_dir=Z:/4.my_work/9.zy
--output_dir=Z:/4.my_work/9.zy/test
--function=matting
```

## 按类别筛选，单数据集处理输入指令参考
```
ccdt
--input_datasets="[{'format': 'labelme','images_dir': 'Z:/4.my_work/9.zy/00/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/00/01.labelme'},]"
--input_dir=Z:/4.my_work/9.zy
--output_dir=Z:/4.my_work/9.zy/filter
--name_classes="['call','call_fuzzy','call2']"
--type_shapes="['rectangle']"
--function=filter
```

## 按类别筛选，多数据集处理输入指令参考
```
ccdt
--input_datasets="[{'format': 'labelme','images_dir': 'Z:/4.my_work/9.zy/00/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/00/01.labelme'},{'format': 'labelme','images_dir': 'Z:/4.my_work/9.zy/11/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/11/01.labelme'},]"
--input_dir=Z:/4.my_work/9.zy
--output_dir=Z:/4.my_work/9.zy/filter
--name_classes="['call','call_fuzzy','call2']"
--type_shapes="['rectangle']"
--function=filter
```

