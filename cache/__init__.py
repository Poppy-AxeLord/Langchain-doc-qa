# cache模块初始化文件
from .pdf_cache import get_pdf_file_hash, save_cache_pdf_topic, load_pdf_topic_from_cache
from .vector_cache import save_vector_db_hash, load_vector_db_hash
from .qa_cache import init_qa_cache, get_qa_cache, set_qa_cache, clear_qa_cache, get_relevance_cache, set_relevance_cache, clear_relevance_cache

__all__ = [
    'get_pdf_file_hash',
    'save_cache_pdf_topic',
    'load_pdf_topic_from_cache',
    'save_vector_db_hash',
    'load_vector_db_hash',
    'init_qa_cache',
    'get_qa_cache',
    'set_qa_cache',
    'clear_qa_cache',
    'get_relevance_cache',
    'set_relevance_cache',
    'clear_relevance_cache'
]
