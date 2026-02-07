import os

def save_vector_db_hash(pdf_hash, flag_path="./chroma_db/pdf_hash.txt"):
    """
    保存向量库对应的PDF哈希值，用于判断是否需要重建向量库
    :param pdf_hash: PDF文件的哈希值
    :param flag_path: 哈希值保存路径
    """
    try:
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as f:
            f.write(pdf_hash)
    except Exception as e:
        print(f"⚠️ 保存向量库哈希失败：{e}")

def load_vector_db_hash(flag_path="./chroma_db/pdf_hash.txt"):
    """
    加载向量库对应的PDF哈希值
    :param flag_path: 哈希值保存路径
    :return: 保存的哈希值（空字符串表示无保存）
    """
    try:
        if not os.path.exists(flag_path):
            return ""
        with open(flag_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ 加载向量库哈希失败：{e}")
        return ""
