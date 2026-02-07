# layered_memory.py
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json

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

class LayeredMemoryManager:
    """分层记忆管理器：短期记忆（原始对话）+ 长期记忆（结构化蒸馏）"""
    def __init__(self, llm, short_term_max_rounds: int = 3):
        self.short_term = ChatMessageHistory()  # 恢复使用ChatMessageHistory，更贴合原生生态
        self.short_term_max_rounds = short_term_max_rounds
        self.long_term: Optional[StructuredMemory] = None
        self.llm = llm
        self.distill_prompt = self._build_distill_prompt()

    def _build_distill_prompt(self) -> ChatPromptTemplate:
        """构建蒸馏结构化信息的Prompt（修复{}转义问题）"""
        return ChatPromptTemplate.from_messages([
            ("system", """请从以下对话中提取结构化核心信息，严格按照JSON格式输出（不要额外解释，仅输出JSON字符串）：
            1. entities：仅提取文档相关的核心实体/术语，不超过5个
            2. user_intents：仅提取用户的核心提问意图，不超过3个
            3. key_conclusions：仅提取文档中的关键结论/回答要点，不超过3个
            4. context_ref：提取对话中的模糊指代（如'它'/'这个'），映射到具体实体
            示例输出格式（仅作参考，需根据实际对话生成）：
            {{{{
                "entities": ["概念1", "概念2"],
                "user_intents": ["意图1"],
                "key_conclusions": ["结论1"],
                "context_ref": {{"它": "概念1"}}
            }}}}"""),  # 关键：所有普通{}转义为{{}}
            ("human", "对话内容：{conversation_text}")  # 这里的{conversation_text}是真实变量，不转义
        ])

    def add_message(self, role: str, content: str):
        """添加消息到短期记忆，并触发蒸馏逻辑"""
        if role == "user":
            self.short_term.add_user_message(content)
        elif role == "assistant":
            self.short_term.add_ai_message(content)
        
        total_messages = len(self.short_term.messages)
        if total_messages >= 2 * self.short_term_max_rounds:
            self._distill_to_long_term()
            self.short_term.messages = self.short_term.messages[-2:]

    def _distill_to_long_term(self):
        """将短期记忆蒸馏为结构化信息，存入长期记忆"""
        conversation_text = ""
        for msg in self.short_term.messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            conversation_text += f"{role}：{msg.content}\n"
        
        try:
            chain = self.distill_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"conversation_text": conversation_text})
            
            # 解析前先清理可能的多余字符（如换行、空格）
            distill_str = response.strip()
            # 若输出包含示例中的外层{{}}，先去除（可选，增强兼容性）
            if distill_str.startswith("{{") and distill_str.endswith("}}"):
                distill_str = distill_str[1:-1]  # 去除首尾各一个大括号，恢复为{}
            
            distill_data = json.loads(distill_str)
            new_struct_mem = StructuredMemory(**distill_data)
            
            if self.long_term is None:
                self.long_term = new_struct_mem
            else:
                self.long_term.entities = list(set(self.long_term.entities + new_struct_mem.entities))[:5]
                self.long_term.user_intents = list(set(self.long_term.user_intents + new_struct_mem.user_intents))[:3]
                self.long_term.key_conclusions = list(set(self.long_term.key_conclusions + new_struct_mem.key_conclusions))[:3]
                self.long_term.context_ref.update(new_struct_mem.context_ref)
            
            print(f"✅ 短期记忆蒸馏完成，长期记忆已更新")
            # ========== 新增：打印长期记忆的结构化内容 ==========
            print(f"\n📋 当前长期记忆（蒸馏后结构化内容）：")
            print(f"   核心实体：{self.long_term.entities}")
            print(f"   用户意图：{self.long_term.user_intents}")
            print(f"   核心结论：{self.long_term.key_conclusions}")
            print(f"   指代映射：{self.long_term.context_ref}")
            # ==================================================
        except json.JSONDecodeError as e:
            print(f"⚠️ 蒸馏结果JSON解析失败：{e}，响应内容：{response}")
        except Exception as e:
            print(f"⚠️ 蒸馏失败：{e}，跳过本次蒸馏")

    def get_combined_memory(self) -> str:
        """获取组合记忆：长期结构化记忆 + 短期原始记忆"""
        long_term_text = self.long_term.to_prompt_text() if self.long_term else ""
        
        short_term_text = "\n最新对话：\n"
        for msg in self.short_term.messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            short_term_text += f"{role}：{msg.content}\n"
        
        return long_term_text + short_term_text

    def clear_short_term(self):
        self.short_term.clear()
        print("🧹 短期记忆已清空")

    def clear_long_term(self):
        self.long_term = None
        print("🧹 长期记忆已清空")

    def clear_all(self):
        self.clear_short_term()
        self.clear_long_term()
        return "🧹 所有记忆已清空"

    @property
    def short_term_messages(self):
        return self.short_term.messages