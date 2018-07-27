<h2>调试过程中遇到的问题和注意事项</h3></br>
<h3>错误点</h3></br>

<ul>
	<li>在GPU上运行时需要设置model = model.cuda()</li></br>
	<li>统计correct个数时需使用item()转换为python数值 即correct = (predicted == labels).sum().item()</li></br>
</ul>
	
<h3>学习到的地方</h3></br>
<ul>
	<li>linux安装CUDA教程 https://blog.csdn.net/u012235003/article/details/54575758</li>
	<li>验证或测试时得到的输出并不是概率，需要通过softmax处理(probility_list = t.nn.functional.softmax(outputs)[:,0].tolist())</li></br>
	<li>zip（）将两个列表在同一维度的元素合并为一个元祖,可用于合并图像编号和对应概率</br>
        如a =[1,2,3]  b=[4,5,6] </br>
        zip(a,b)=[(1,4),(2,5),(3,6)]</br>
	</li>
	<li>模型的保存，可继承于t.nn.Module,重写一个类，定义save 和load方法</br>
		load: self.load_state_dict(t.load(path))  #path为输入的参数</br>
		save: t.save(self.state_dict(),name)      #name为保存的模型名称</br>
	</li>
	
</ul>