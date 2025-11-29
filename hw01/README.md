第一次课的lab居然是cs188的经典，老师好品位

按照题目要求目前仅仅完成了q1-q3

 - 我们啥比conda环境，使用以下命令比特级还原环境：
```bash
# 1. 新建空环境并直接按 URL 安装
conda env create -n cs188_submit --file environment.lock

# 2. 激活即用
conda activate cs188_submit

# 3. 验证
python -c "import torch, numpy, matplotlib; print(torch.__version__)"
```

 - 如果不行的话试试这个：
```bash
# 1. 让 conda 自己解析依赖
conda env create -n cs188_loose --file environment.yml

# 2. 激活
conda activate cs188_loose

# 3. 验证
python -c "import torch, numpy, matplotlib; print(torch.__version__)"
```
换源自己解决，其他依赖自己解决。依赖中的intel-openmp和mkl和mkl-service也许不适合所有人安装


有mamba可以试试，我是用mamba的。micromamba之类的不知道行不行