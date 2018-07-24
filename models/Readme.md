模型定义的<h1>注意事项</h1>：
尽量使用nn.Sequential（比如AlexNet）
将经常使用的结构封装成子Module（比如GoogLeNet的Inception结构，ResNet的Residual Block结构）
将重复且有规律性的结构，用函数生成（比如VGG的多种变体，ResNet多种变体都是由多个重复卷积层组成）