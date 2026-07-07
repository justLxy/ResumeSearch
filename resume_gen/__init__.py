"""确定性 + LLM 缓存的模拟简历生成包。

对外主要入口：
- generate(count, seed, use_llm=False) -> list[dict]
- SEED
- _quality_stats(docs)
- main()
"""
from resume_gen.builder import SEED, _quality_stats, generate, main

__all__ = ["generate", "SEED", "_quality_stats", "main"]
