def normalize_text(text):
    """
    标准化文本，去除多余的空格和换行
    :param text: 原始文本
    :return: 标准化后的文本
    """
    return ' '.join(text.strip().split())
