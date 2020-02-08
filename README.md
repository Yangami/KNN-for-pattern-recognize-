# KNN-for-pattern-recognize
-----------
[功能描述]股票形态识别（如W双底）用图像识别的方法准确率高但速度慢（因要画图），用K-近邻方法以数值型数据计算快准确率基本符合要求(查准率70%左右)，可用于对决策时间有要求的交易。 
工作完成情况：1、W双底识别模型查准确率约70% 2、模型文件上载到聚宽后可在回测中调用 
优化方向：该模型识别时间短但训练时间长，这是由于训练过程不能完全自动化要分流程进行。优化训练过程使模型准确率尽可能快得提高。
[开发环境]Python3.6 
[项目结构简介及运行说明]见：辅助说明文件