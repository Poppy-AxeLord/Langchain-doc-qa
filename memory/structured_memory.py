from pydantic import BaseModel, Field
from typing import List, Dict

class StructuredMemory(BaseModel):
    """长期记忆的结构化数据模板（蒸馏结果）"""
    entities: List[str] = Field(description="对话中涉及的核心实体，如文档中的概念、术语、产品名")
    user_intents: List[str] = Field(description="用户的核心意图，如询问定义、对比差异、求解问题")
    key_conclusions: List[str] = Field(description="对话中的核心结论/文档中的关键信息")
    context_ref: Dict[str, str] = Field(description="指代映射，如'它'→'某个概念'")

    def to_dict(self) -> Dict:
        return {
            "entities": self.entities,
            "user_intents": self.user_intents,
            "key_conclusions": self.key_conclusions,
            "context_ref": self.context_ref
        }

    def to_prompt_text(self) -> str:
        return f"""
        历史对话核心信息：
        1. 核心实体：{', '.join(self.entities)}
        2. 用户意图：{', '.join(self.user_intents)}
        3. 核心结论：{', '.join(self.key_conclusions)}
        4. 指代映射：{'; '.join([f'{k}→{v}' for k, v in self.context_ref.items()])}
        """
