# Python代码注释检查器

程序员使用注释来屏蔽那些自己不想运行的代码。但是在提交代码时，应当把这些无用的代码段彻底删除。这是一种良好的编程习惯。

`codeclf`是一个检查指定目录下所有Python代码（`*.py`）中的注释（`#`开头的所有行）是否含有代码的小工具。

利用`codeclf`，你可以清理自己项目中无用的、可能引起错误的注释。

支持命令行调用。使用命令行调用的时候，第一个参数是你要检查的项目目录。

`-mc`参数是“逐字符”模型的位置，`-mt`参数是“分词”模型的位置，`-v`参数是单词表的位置，他们都有默认值。

运行结果为`code_warning.json`文件，囊括了`codeclf`认为有可能是代码的所有注释行。

使用前，请先验证自己是否安装好了`Python3`、`tensorflow`。我在windows 10下测试没有问题。

`tensorflow`版本：`2.3.0`，安装`tensorflow`需要使用 `Python 3.5-3.8`、`pip 19.0` 及更高版本

如对`tensorflow`安装上有疑惑，请参考[tensorflow安装指南](https://www.tensorflow.org/install/pip?hl=zh-cn)。

使用样例：
```bash
python codeclf.py "." // 检查本项目中注释存在的问题

python codeclf.py "/home/user/tensorflow-master" // 检查/home/user/tensorflow-master文件夹下的所有python代码

python codeclf.py "." -mc "models/mc.hdf5" -mt "models/mt_20000.hdf5" -v "vocabs/vocab_20000.txt" // 指定逐字符模型的路径（参数-mc）与词法分析模型路径（参数-mt）与字典表路径（参数-v）用于注释分类。这三个参数即便不指定，默认就是这样执行的
```

一旦程序认为检测目录内包含被注释掉的代码行，会在`results`文件夹下生成`code_warining.json`文件，包含包含有代码注释行的文件名、行号和具体内容等信息。

如果生成了`code_warining.json`文件，通过运行`python parsejson.py`即可将`code_warning.json`转为`code_warning.csv`方便使用excel打开。

如果有任何问题，请联系`bjutzyt (at) 126.com`