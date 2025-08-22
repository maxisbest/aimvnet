

## aimvnet 项目说明
- 用于动作捕捉和评估打分
- 适用于MacOS M芯片系列

## Python 3.11 虚拟环境的安装方法:
```commandline
# 1. 安装Homebrew（如果还没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装Python 3.11; Python is installed as: /opt/homebrew/bin/python3.11
brew install python@3.11

# 3. 使用特定版本创建虚拟环境(如果是Homebrew安装的话)
/opt/homebrew/bin/python3.11 -m venv movenet-env

# 4. 激活环境(这个环境就在home目录(~)下.
source movenet-env/bin/activate

# 5. 验证Python版本
python --version # 应该显示Python 3.11.x
```

## TensorFlow相关依赖包安装命令:
```commandline
# 环境安装文件 requirements.txt 可以在项目仓库中找到
pip install -r requirements.txt 
```

