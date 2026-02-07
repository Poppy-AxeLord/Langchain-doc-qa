import os

def ensure_directory(directory):
    """
    确保目录存在，如果不存在则创建
    :param directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
