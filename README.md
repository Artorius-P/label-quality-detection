# label-quality-detection
工业中印刷标签质量检测与识别系统

## Reader模块

### 属性

| 属性名 | 类型 | 备注 |
| :---- | :---- | :---- |
| img | numpy.uint8 | 用来识别的图片 |
| result | String | 从图像读出的结果 |

### 方法

| 方法名 | 输入 | 输出 | 备注 |
| :---- | :---- | :---- | :---- |
| read_from_file | path:Sting | is_qualified: bool | 从文件读取图片并输出检测结果 |
| read_from_camera | void | is_qualified: bool | 从摄像头拍摄图片并输出检测结果 |

## Detector模块

Detector模块负责图像的检测，包含如下功能：

- 分割图像中需要进行质量检测和识别的子图像区域。
- 标签质量检测功能，一个合格的标签要求与近邻周边有一定的间距，如果间距太小，则视为不合格产品。
- 一维码标签的识别功能。
- 二维码标签的识别功能。
- 标签周围光学字符识别功能。


