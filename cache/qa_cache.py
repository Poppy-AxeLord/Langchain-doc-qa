from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

# 全局缓存对象
_qa_cache = {}
_relevance_cache = {}
_llm_cache = None

def init_qa_cache():
    """
    初始化QA缓存
    """
    global _qa_cache, _relevance_cache, _llm_cache
    _qa_cache = {}          # 缓存最终问答结果：key=标准化问题，value=(回答, 来源)
    _relevance_cache = {}   # 缓存相关性判断：key=标准化问题，value=(是否相关, 理由)
    _llm_cache = InMemoryCache()
    set_llm_cache(_llm_cache)

def get_qa_cache():
    """
    获取QA缓存
    :return: QA缓存字典
    """
    global _qa_cache
    return _qa_cache

def set_qa_cache(key, value):
    """
    设置QA缓存
    :param key: 缓存键
    :param value: 缓存值
    """
    global _qa_cache
    _qa_cache[key] = value

def clear_qa_cache():
    """
    清空QA缓存
    """
    global _qa_cache
    _qa_cache.clear()

def get_relevance_cache():
    """
    获取相关性缓存
    :return: 相关性缓存字典
    """
    global _relevance_cache
    return _relevance_cache

def set_relevance_cache(key, value):
    """
    设置相关性缓存
    :param key: 缓存键
    :param value: 缓存值
    """
    global _relevance_cache
    _relevance_cache[key] = value

def clear_relevance_cache():
    """
    清空相关性缓存
    """
    global _relevance_cache
    _relevance_cache.clear()

def clear_all_cache():
    """
    清空所有缓存
    """
    clear_qa_cache()
    clear_relevance_cache()
    if _llm_cache:
        _llm_cache.clear()
