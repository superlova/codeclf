# Python代码注释检查器

程序员使用注释来屏蔽那些自己不想运行的代码。但是在提交代码时，应当把这些无用的代码段彻底删除。这是一种良好的编程习惯。

`codeclf`是一个检查指定目录下所有Python代码（`*.py`）中的注释（`#`开头的所有行）是否含有代码的小工具。

利用`codeclf`，你可以清理自己项目中无用的、可能引起错误的注释。

支持命令行调用。使用命令行调用的时候，第一个参数是你要检查的项目目录。

`-mc`参数是“逐字符”模型的位置，`-mt`参数是“分词”模型的位置，`-v`参数是单词表的位置，他们都有默认值。

运行结果为`code_warning.json`文件，囊括了`codeclf`认为有可能是代码的所有注释行。

使用前，请先验证自己是否安装好了`python3`、`tensorflow`与`numpy`。我在windows 10下测试没有问题。

使用样例：
```bash
python codeclf.py "." // 检查本项目中注释存在的问题

python codeclf.py "." -mc "models/mc.hdf5" -mt "models/mt_20000.hdf5" -v "vocabs/vocab_20000.txt"
```
如果有任何问题，请联系bjutzyt (at) 126.com