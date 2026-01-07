import os
import hashlib
import json

def get_pdf_file_hash(pdf_path):
    """
    生成PDF文件的唯一MD5哈希值，用于判断文件是否修改
    :param pdf_path: PDF文件的路径
    :return: 文件的MD5哈希字符串（空字符串表示文件不存在）
    """
    if not os.path.exists(pdf_path):
        return ""
    try:
        with open(pdf_path, "rb") as f:
            file_content = f.read()
            file_hash = hashlib.md5(file_content).hexdigest()
        return file_hash
    except Exception as e:
        print(f"⚠️ 生成PDF哈希失败：{e}")
        return ""

def save_cache_pdf_topic(pdf_hash, topic_info, cache_path="./cache_db/pdf_topic_cache.json"):
    """
    将PDF主题信息缓存到本地JSON文件
    :param pdf_hash: PDF文件的哈希值（作为缓存键）
    :param topic_info: 提取的PDF主题信息（字典）
    :param cache_path: 缓存文件路径
    """
    try:
        # 读取现有缓存
        cache_data = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        # 更新缓存
        cache_data[pdf_hash] = topic_info
        # 写入文件（确保目录存在）
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 缓存PDF主题失败：{e}")

def load_pdf_topic_from_cache(pdf_hash, cache_path="./cache_db/pdf_topic_cache.json"):
    """
    从本地缓存文件加载PDF主题信息
    :param pdf_hash: PDF文件的哈希值
    :param cache_path: 缓存文件路径
    :return: 缓存的主题信息（字典），None表示无缓存
    """
    try:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        return cache_data.get(pdf_hash)
    except Exception as e:
        print(f"⚠️ 加载PDF主题缓存失败：{e}")
        return None

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