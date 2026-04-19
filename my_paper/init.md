当前仓库是robometer (original_paper_robometer) 的代码仓库，我尝试在它的基础上做my_paper。每次对话之前你都要看一下下述文件：

setup.md 是环境配置文档
paper-draft-v2.md 是论文正文草稿
experiment-section-v2.md 是实验部分草稿，不是最终稿，实际的文字要匹配我实际做的实验
experiment-design-v2.md 是我打算要做的实验

当前进度：
- python环境，数据 preprocess, 模型下载完成
- Dry run 通过
- unsloth 下的模型文件是从huggginface cache里的对应地方链接过来的

bashrc 中的内容：

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export ROBOMETER_DATASET_PATH=/root/autodl-tmp/raw_datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/root/autodl-tmp/processed_datasets