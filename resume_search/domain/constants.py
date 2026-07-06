"""领域词表：学历、院校档位、技能标签的规范化映射。

这些是"业务语义"常量（同义词、显示名、排序），区别于 config.py 的可调参数。
被 normalization / filters / facets 等 service 引用。
"""
from __future__ import annotations

DEGREE_ALIASES = {
    "博士研究生": "博士",
    "博士": "博士",
    "硕士研究生": "硕士",
    "硕士": "硕士",
    "学士": "本科",
    "本科": "本科",
}
DEGREE_ORDER = ("本科", "硕士", "博士")

# 院校档位 key → UI 显示名。"其他" 不在名单里，用补集实现，不出现在此表。
SCHOOL_TIER_LABELS = {
    "985": "985",
    "211": "211",
    "双一流": "双一流",
    "c9": "C9",
    "qs50_overseas": "海外QS50",
}
SCHOOL_TIER_OTHER = "其他"
# 兼容前端/LLM 可能传入的显示名或别名 → 规范 key。
SCHOOL_TIER_ALIASES = {
    "985": "985",
    "211": "211",
    "双一流": "双一流",
    "c9": "c9",
    "C9": "c9",
    "qs50_overseas": "qs50_overseas",
    "海外QS50": "qs50_overseas",
    "海外qs50": "qs50_overseas",
    "海外": "qs50_overseas",
    "其他": SCHOOL_TIER_OTHER,
}
CANONICAL_SKILL_LABELS = {
    "c": "C",
    "c++": "C++",
    "c#": "C#",
    "css": "CSS",
    "docker": "Docker",
    "html": "HTML",
    "java": "Java",
    "javascript": "JavaScript",
    "jvm": "JVM",
    "linux": "Linux",
    "mysql": "MySQL",
    "nlp": "NLP",
    "pytorch": "PyTorch",
    "redis": "Redis",
    "spark": "Spark",
    "spring boot": "Spring Boot",
    "spring cloud": "Spring Cloud",
    "sql": "SQL",
    "tensorflow": "TensorFlow",
    "typescript": "TypeScript",
    "vue": "Vue",
}
