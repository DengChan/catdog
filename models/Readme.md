<h3>模型定义的注意事项：</h3>

<ul>
<li>尽量使用nn.Sequential（比如AlexNet）</li>
<li>将经常使用的结构封装成子Module（比如GoogLeNet的Inception结构，ResNet的Residual Block结构）</li>
<li>将重复且有规律性的结构，用函数生成（比如VGG的多种变体，ResNet多种变体都是由多个重复卷积层组成</li>
</ul>


