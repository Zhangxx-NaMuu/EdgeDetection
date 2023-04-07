# 3D 颈缘线识别



## 	文件树

.

├── Example.stl			 —— 三角网格输入文件

├── README.md		   —— README

├── main.py				   —— 主函数文件

└── requirements.txt	 —— pip依赖项

### 安装

`pip install -r requirements.txt`

或者

`conda env create -f environment.yml`

### 运行

`python main.py`

### 结果

控制台

![image-20210817113843223](README.assets/image-20210817113843223.png)

窗口一

![image-20210817113802447](README.assets/image-20210817113802447.png)

窗口二

![image-20210817113824829](README.assets/image-20210817113824829.png)

## 设计思路

1. 边检测算法；

   ![image-20210817134100280](README.assets/image-20210817134100280.png)

2. 多次RANSAC算法去掉多余点；

![image-20210817134137266](README.assets/image-20210817134137266.png)

3. 根据平均曲率匹配特征点；

   ![image-20210817134532195](README.assets/image-20210817134532195.png)

4. 体素降采样和离群点去噪声。

   ![image-20210817134421081](README.assets/image-20210817134421081.png)