<div align="center">
  <img src="resources/LOGO.png" width="450"/>
  <center>
  
  [![PyPI](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/cpdt/)
  [![license](https://img.shields.io/badge/license-GNU%20General%20Public%20License%20v3-green)](https://github.com/540717421/chipeak_data_tool)
  
  </center>
</div>

## Introduction

[简体中文](README.md) | English

&nbsp;&nbsp;&nbsp;&nbsp;chipeak_ cv_ data_ tool is mainly used for the conversion between various data sets (computer vision) formats, such as coco, label me, etc. In addition, it also integrates common processing functions of data, files, videos and other files.


## Major features

- **Modular Design**

  Various data sets are decoupled into different module components. By combining different module components, users can easily build custom data set processing.

## Installation

pip install cpdt

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of chipeak_cv_data_tool.

## Related annotation tools
labelme:https://github.com/wkentaro/labelme

## Implementation of command line function
* Data set formats are not supported 
    - [x] Labelme format dataset https://github.com/wkentaro/labelme
    - [x] Coco format dataset
* Data processing function 
    - [x] Conversion of dataset dimension files to each other
    - [x] Save the intercepted target as an image (currently supported: rectangular box,)
* Image (file) processing 
    - [x] Divide data by proportion or number of files (e.g. file allocation or training test set division)
    - [x] Coco format dataset
* Video processing 
    - [x] Video slicing into images
    - [x] Image synthesis into video

## Function realization of Python module
* Data set formats are not supported 
    - [x] Labelme format dataset https://github.com/wkentaro/labelme
    - [x] Coco format dataset
* Data processing function 
    - [x] Conversion of dataset dimension files to each other
    - [x] Save the intercepted target as an image (currently supported: rectangular box,)
* Image (file) processing 
    - [x] Divide data by proportion or number of files (e.g. file allocation or training test set division)
    - [x] Coco format dataset
* Video processing 
    - [x] Video slicing into images
    - [x] Image synthesis into video
    
## License

This project is released under the [GNU General Public License v3.0 2.0](LICENSE).

## Changelog

v1.0 was released in 08/11/2021. 

Supported datasets:

- [x] Coco dataset
- [x] Via annotation dataset
- [x] Labelme dimension dataset


Supported features:

- [x] Labelme dataset matting
- [x] Coco dataset to labelme dataset
- [x] Labelme dataset to coco dataset
- [x] Filter the labelme dataset by label category

## Contributing

no

## Acknowledgement

no

## Citation

```
@misc{ccdt2021,
  author =       {Zhan Yong},
  title =        {{ccdt: Image Polygonal Annotation with Python}},
  howpublished = {url{https://github.com/chipeak/chipeak_cv_data_tool.git}},
  year =         {2021}
}
```

## Projects in ChiPeak

- [ccdt](https://github.com/540717421/chipeak_data_tool): chipeak_ cv_ data_ Tool AI data processing toolbox

## Welcome to the chipeak community

Scan the QR code below to follow the [official website] of Xinfeng technology team（ http://http://www.chipeak.com/ ）, join the [official exchange] of Xinfeng technology team（ http://www.chipeak.com/account/login )
<div align="center">
<img src="/resources/xf_rq_code.png" height="200" />
</div>
We will work for you in the chipeak community

- 📢 Share the cutting-edge core technologies of AI framework
- 💻 Interpretation of CCDT common module source code
- 📰 Release news about chipeak
- 🚀 This paper introduces the cutting-edge algorithm developed by chipeakdetection
- 🏃 Get more efficient question answering and feedback
- 🔥 Provide a platform for full communication with developers from all walks of life

Dry goods are full 📘， Wait for you 💗， The ChiPeak community is looking forward to your participation 👬

## Security
[![Security Status](https://www.murphysec.com/platform3/v31/badge/1673251865674670080.svg)](https://www.murphysec.com/console/report/1673251865636921344/1673251865674670080)
