"""从生成数据的 ground-truth 派生 eval_queries.jsonl 的相关性标注（qrels）。

可信度原则（见 PROJECT_REVIEW 第十节的讨论）：
- 不让 LLM 既出题又判分（同源偏差）。
- 不靠跑检索系统反推相关集（循环论证）。
- 利用 generate_mock_resumes 的确定性：每份简历的岗位族标签就是 ground-truth，
  qrels 由"族归属 + 结构化字段"客观派生，可复现。

分级口径：
- grade 3（核心相关）：与 query 主题同族的简历。
- grade 2（明显相关）：技能/语义高度重叠的相邻族，由 _FAMILY_ADJACENCY 按实测
  技能重叠 + 岗位常识派生（topical 类 query 自动展开；实体/专业/精确类不适用）。
- forbidden：与 query 主题"沾边但不对"的 hard-negative 族（不该进前排）。
- 库外负例（negative_semantic）：库里压根没有的主题 → expect_empty=true。

用法：
    python build_eval_queries.py            # 写 eval_queries.jsonl
    python build_eval_queries.py --dry-run  # 只打印统计，不写文件
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import generate_mock_resumes as gen

PRIMARY_METRICS = [
    "P@5", "P@10", "R@10", "R@50", "R@100", "MRR@10", "NDCG@5", "NDCG@10", "forbidden@10",
]
GRADE_WEIGHTS = {"core": 3, "related": 2, "weak_related": 1, "negative": 0}


def _load_ground_truth() -> tuple[dict[str, list[str]], dict[str, dict]]:
    """复现确定性生成，返回 family→[resume_id] 和 resume_id→doc。"""
    docs = gen.generate(200, gen.SEED)
    by_family: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, dict] = {}
    for d in docs:
        by_family[d["_family"]].append(d["resume_id"])
        by_id[d["resume_id"]] = d
    return dict(by_family), by_id


def _evaluation_block() -> dict[str, Any]:
    return {"grade_weights": GRADE_WEIGHTS, "primary_metrics": PRIMARY_METRICS}


# ---------------------------------------------------------------------------
# 语义类 query 规格：query 文本 + 核心族(grade3) + 相邻族(grade2) + forbidden 族
# 相邻族(grade2)反映"技能重叠但不是同一主题"的合理弱相关；forbidden 用 hard-neg 族。
# ---------------------------------------------------------------------------
SEMANTIC_SPECS: list[dict[str, Any]] = [
    {
        "id": "sem_enterprise_rag", "query": "找做过企业知识库 RAG、向量检索和大模型问答落地的人",
        "expected_lexical": "RAG 向量检索",
        "core": ["ml_llm"], "forbidden_families": ["hn_search_ops"],
    },
    {
        "id": "sem_blue_team_hunting", "query": "需要蓝队应急响应，能做日志取证、威胁狩猎和 ATT&CK 分析",
        "expected_lexical": "应急响应 威胁狩猎",
        "core": ["blue_team"], "forbidden_families": ["hn_it_ops"],
    },
    {
        "id": "sem_kernel_fuzzing", "query": "安全研究要会内核漏洞挖掘、Fuzzing、符号执行和 0day Exploit",
        "expected_lexical": "漏洞挖掘 Fuzzing 符号执行",
        "core": ["security_research"], "forbidden_families": [],
    },
    {
        "id": "sem_java_arch_finance", "query": "找能做金融级 Java 高并发架构、分库分表、分布式事务和 JVM 调优的人",
        "expected_lexical": "Java 分库分表 分布式事务",
        "core": ["backend_java"], "forbidden_families": [],
    },
    {
        "id": "sem_go_cloud_native", "query": "需要 Go 云原生底层能力，做过 Kubernetes Operator、服务网格和高并发网关",
        "expected_lexical": "Go Kubernetes 服务网格",
        "core": ["backend_go"], "forbidden_families": [],
    },
    {
        "id": "sem_frontend_3d", "query": "前端候选要能做 3D 可视化、WebGL 地图引擎、大屏交互和 shader 效果",
        "expected_lexical": "WebGL shader",
        "core": ["frontend"], "forbidden_families": [],
    },
    {
        "id": "sem_low_latency_cpp", "query": "C++ 候选要有超低延迟、高频交易、RDMA、内核旁路和无锁编程经验",
        "expected_lexical": "C++ RDMA 无锁编程",
        "core": ["backend_cpp"], "forbidden_families": ["hn_cpp_biz"],
    },
    {
        "id": "sem_growth_data", "query": "数据分析希望做过 A/B 实验、指标体系、归因分析、留存提升和电商增长",
        "expected_lexical": "指标体系 归因分析 留存提升",
        "core": ["data_analysis"], "forbidden_families": ["hn_bi_report"],
    },
]

# 多技能组合：core 同族，forbidden 用对应 hard-neg
SKILL_COMBO_SPECS: list[dict[str, Any]] = [
    {"id": "skill_rag_langchain_pytorch", "query": "Python PyTorch RAG LangChain",
     "core": ["ml_llm"], "forbidden_families": ["hn_search_ops"]},
    {"id": "skill_go_k8s_grpc", "query": "Golang Kubernetes gRPC Redis",
     "core": ["backend_go"], "forbidden_families": []},
    {"id": "skill_vue3_ts_webgl", "query": "Vue3 TypeScript WebGL 前端性能优化",
     "core": ["frontend"], "forbidden_families": []},
    {"id": "skill_sql_python_abtest", "query": "SQL Python A/B测试 数据可视化",
     "core": ["data_analysis"], "forbidden_families": ["hn_bi_report"]},
    {"id": "skill_devsecops_sast", "query": "DevSecOps SAST IAST Docker",
     "core": ["devsecops"], "forbidden_families": []},
    {"id": "skill_java_spring_mq", "query": "Java Spring Cloud Redis MySQL RocketMQ",
     "core": ["backend_java"], "forbidden_families": []},
    {"id": "skill_cpp_rdma_lockfree", "query": "C++ RDMA 无锁编程 Linux",
     "core": ["backend_cpp"], "forbidden_families": ["hn_cpp_biz"]},
    {"id": "skill_pentest_redteam", "query": "内网渗透 Cobalt Strike 横向移动 提权",
     "core": ["red_team"], "forbidden_families": []},
]

# 长 JD 匹配：core 同族。expected_lexical 只列"期望被压缩保留的核心检索词"，
# 用于 planner 子集判定（不要求 LLM 逐字复现原句，但核心词不应丢）。
JD_SPECS: list[dict[str, Any]] = [
    {"id": "jd_rag_llm", "core": ["ml_llm"], "forbidden_families": ["hn_search_ops"],
     "expected_lexical": "LLM RAG 向量检索 Python PyTorch LangChain",
     "query": "岗位：LLM/RAG 应用工程师。职责：负责企业知识库问答、文档解析、向量检索、召回排序、Prompt 设计和模型微调，能用 Python、PyTorch、LangChain 做工程落地。要求：熟悉 RAG 评测、长文本处理和业务系统集成，有 ToB 知识库项目经验优先。"},
    {"id": "jd_devsecops", "core": ["devsecops"], "forbidden_families": [],
     "expected_lexical": "DevSecOps SAST SCA IAST 容器",
     "query": "岗位：DevSecOps 安全平台架构师。职责：建设 SAST、SCA、IAST 检测能力，集成 CI/CD 流水线安全卡点，做容器与镜像安全。要求：熟悉软件供应链安全、容器安全，掌握 Java 或 Go。"},
    {"id": "jd_java_finance", "core": ["backend_java"], "forbidden_families": [],
     "expected_lexical": "Java 分布式事务 分库分表 JVM Spring Cloud",
     "query": "岗位：高级 Java 架构师。职责：负责金融核心链路高并发、高可用、分布式事务、分库分表和 JVM 调优。要求：精通 Spring Cloud 微服务，熟悉 RocketMQ/Kafka 与 Redis。"},
    {"id": "jd_go_cloud", "core": ["backend_go"], "forbidden_families": [],
     "expected_lexical": "Go Kubernetes gRPC 服务网格",
     "query": "岗位：Go 云原生基础设施工程师。职责：建设 Kubernetes Operator、服务网格、高并发网关与可观测体系。要求：精通 Go 与 gRPC，熟悉 etcd、Prometheus 和容器编排。"},
    {"id": "jd_blue_team", "core": ["blue_team"], "forbidden_families": ["hn_it_ops"],
     "expected_lexical": "应急响应 日志取证 威胁狩猎 ATT&CK SIEM",
     "query": "岗位：蓝队应急响应专家。职责：负责企业安全事件响应、日志取证、勒索软件溯源、流量分析与威胁狩猎。要求：熟悉 ATT&CK、SIEM、EDR，掌握 Python 自动化处置。"},
    {"id": "jd_red_team", "core": ["red_team"], "forbidden_families": [],
     "expected_lexical": "红队 内网横向移动 C2 内网渗透",
     "query": "岗位：红队攻防专家。职责：负责红队演练、AD 域渗透、内网横向移动、C2 通信和权限维持。要求：熟悉 Cobalt Strike、内网渗透，掌握免杀与对抗技术。"},
    {"id": "jd_frontend_3d", "core": ["frontend"], "forbidden_families": [],
     "expected_lexical": "WebGL Three.js 可视化 Vue3 TypeScript shader",
     "query": "岗位：高级前端可视化工程师。职责：负责 WebGL、Three.js、大屏交互和数据可视化引擎建设。要求：精通 Vue3 或 React 与 TypeScript，熟悉 shader 与渲染性能优化。"},
    {"id": "jd_cpp_hft", "core": ["backend_cpp"], "forbidden_families": ["hn_cpp_biz"],
     "expected_lexical": "C++ 低延迟 高频交易 无锁 内核旁路 RDMA",
     "query": "岗位：C++ 低延迟系统专家。职责：建设高频交易或实时交易基础设施，优化网络收发、无锁并发与内核旁路。要求：精通 C++，熟悉 RDMA、DPDK 与性能剖析。"},
    {"id": "jd_security_research", "core": ["security_research"], "forbidden_families": [],
     "expected_lexical": "漏洞挖掘 Fuzzing 逆向 符号执行 C++",
     "query": "岗位：高级漏洞研究员。职责：负责浏览器、内核、开源组件漏洞挖掘，建设 Fuzzing 平台并分析利用链。要求：精通 C/C++ 与逆向，熟悉符号执行与漏洞利用。"},
    {"id": "jd_data_growth", "core": ["data_analysis"], "forbidden_families": ["hn_bi_report"],
     "expected_lexical": "A/B 实验 指标体系 归因 留存 SQL Python",
     "query": "岗位：数据分析专家（增长方向）。职责：负责 A/B 实验设计、指标体系建设、归因分析和留存提升。要求：精通 SQL 与 Python，熟悉因果推断与实验平台。"},
    {"id": "jd_testing", "core": ["testing"], "forbidden_families": [],
     "expected_lexical": "自动化测试 性能压测 Python Selenium",
     "query": "岗位：高级测试开发工程师。职责：建设自动化测试平台、接口与 UI 自动化、性能压测与质量度量。要求：掌握 Python，熟悉 Selenium、JMeter 与持续集成。"},
    {"id": "jd_product_security", "core": ["product"], "forbidden_families": [],
     "expected_lexical": "SASE ZTNA 零信任 需求分析 项目管理",
     "query": "岗位：安全产品经理。职责：负责 SASE、ZTNA、SOC 态势感知、零信任访问产品规划与落地。要求：熟悉安全产品体系，具备需求分析与项目管理能力。"},
    {"id": "jd_pentest", "core": ["red_team"], "forbidden_families": [],
     "expected_lexical": "内网渗透 Web 渗透 漏洞挖掘 Burp Metasploit",
     "query": "岗位：渗透测试工程师。职责：负责 Web 与内网渗透、资产测绘、漏洞挖掘与渗透报告。要求：熟悉注入、越权、反序列化漏洞，掌握 Burp Suite 与 Metasploit。"},
]

# 库外负例：库里压根没有这些主题 → 期望返回空
NEGATIVE_SPECS: list[dict[str, Any]] = [
    {"id": "neg_quantum_chip", "query": "量子计算芯片设计与超导量子比特控制经验"},
    {"id": "neg_sap_abap", "query": "SAP ABAP 财务模块实施顾问，熟悉 FICO 和企业 ERP 上线"},
    {"id": "neg_ios_swift_av", "query": "iOS Swift 音视频播放器内核开发，熟悉 AVFoundation 和 Metal"},
    {"id": "neg_medical_dicom", "query": "医学影像放疗计划系统 DICOM 三维重建和临床算法验证"},
    {"id": "neg_slam_robotics", "query": "机器人 SLAM 定位导航、ROS 移动机器人路径规划和传感器融合"},
    {"id": "neg_unity_game", "query": "Unity C# 游戏客户端渲染、热更新和性能优化"},
    {"id": "neg_oracle_dba", "query": "Oracle DBA，负责 RAC 集群、备份恢复、SQL 调优和数据库巡检"},
    {"id": "neg_embedded_rtos", "query": "嵌入式 RTOS 固件开发、单片机驱动和电机控制"},
]


# ---------------------------------------------------------------------------
# case 构造器
# ---------------------------------------------------------------------------
def _semantic_plan(query: str, lexical: str | None = None) -> dict[str, Any]:
    # lexical_query 用核心词（子集判定），semantic_query 保留原句（系统不压缩它）。
    return {"intent": "semantic", "lexical_query": lexical or query, "semantic_query": query,
            "enable_dense": True, "enable_rerank": True}


# 族邻接表：与 core 主题"技能/语义高度重叠但不是同一岗位"的相邻族 → grade 2。
# 依据 generate_mock_resumes 各族技能集的实测重叠（Jaccard）+ 岗位常识，取对称闭包。
# 这反映真实招聘里"不是首选但明显沾边、排进结果合理"的弱相关，让 NDCG 用上分级口径。
# 注意：邻接族都是真实岗位族，与 forbidden 用的 hn_* hard-negative 族不相交，
# 不会让同一文档既相关又被禁。
_FAMILY_ADJACENCY: dict[str, set[str]] = {
    "ml_llm": {"data_analysis"},
    "data_analysis": {"ml_llm"},
    "backend_go": {"backend_java", "devsecops"},
    "backend_java": {"backend_go"},
    "devsecops": {"backend_go"},
    "backend_cpp": {"security_research"},
    "security_research": {"backend_cpp", "red_team"},
    "red_team": {"security_research", "blue_team"},
    "blue_team": {"red_team"},
}


def _related_families(core: list[str], forbidden_families: list[str]) -> list[str]:
    """从邻接表派生 core 的相邻族，排除 core 自身和 forbidden 族。"""
    core_set = set(core)
    forbidden_set = set(forbidden_families)
    related: set[str] = set()
    for f in core:
        related |= _FAMILY_ADJACENCY.get(f, set())
    return sorted(related - core_set - forbidden_set)


def _grade_map(spec: dict, fam: dict[str, list[str]]) -> tuple[dict[str, int], list[str]]:
    """把 core/related 族展开成 relevance dict，forbidden 族展开成 forbidden_ids。

    related 未显式给出时，按 _FAMILY_ADJACENCY 自动派生（grade 2）。
    """
    relevance: dict[str, int] = {}
    for f in spec.get("core", []):
        for rid in fam.get(f, []):
            relevance[rid] = 3
    related = spec.get("related")
    if related is None:
        related = _related_families(spec.get("core", []), spec.get("forbidden_families", []))
    for f in related:
        for rid in fam.get(f, []):
            relevance.setdefault(rid, 2)
    forbidden: list[str] = []
    for f in spec.get("forbidden_families", []):
        forbidden.extend(fam.get(f, []))
    return relevance, sorted(forbidden)


def _build_semantic_cases(fam, _by_id) -> list[dict]:
    cases = []
    for spec in SEMANTIC_SPECS:
        relevance, forbidden = _grade_map(spec, fam)
        cases.append({
            "id": spec["id"], "type": "semantic_capability", "query": spec["query"],
            "expected_plan": _semantic_plan(spec["query"], spec.get("expected_lexical")),
            "relevance": relevance, "forbidden_ids": forbidden,
            "evaluation": _evaluation_block(),
        })
    return cases


def _build_skill_combo_cases(fam, _by_id) -> list[dict]:
    cases = []
    for spec in SKILL_COMBO_SPECS:
        relevance, forbidden = _grade_map(spec, fam)
        cases.append({
            "id": spec["id"], "type": "skill_combo", "query": spec["query"],
            "expected_plan": _semantic_plan(spec["query"]),
            "relevance": relevance, "forbidden_ids": forbidden,
            "evaluation": _evaluation_block(),
        })
    return cases


def _build_jd_cases(fam, _by_id) -> list[dict]:
    cases = []
    for spec in JD_SPECS:
        relevance, forbidden = _grade_map(spec, fam)
        # lexical_query 用 expected_lexical（期望被压缩保留的核心词），子集判定；
        # semantic_query 保留原句（系统对 semantic_query 不做压缩）。
        plan = {"intent": "semantic", "lexical_query": spec["expected_lexical"],
                "semantic_query": spec["query"], "enable_dense": True, "enable_rerank": True}
        cases.append({
            "id": spec["id"], "type": "jd_match", "query": spec["query"],
            "expected_plan": plan,
            "relevance": relevance, "forbidden_ids": forbidden,
            "evaluation": _evaluation_block(),
        })
    return cases


def _build_negative_cases(_fam, _by_id) -> list[dict]:
    cases = []
    for spec in NEGATIVE_SPECS:
        # 负例期望返回空，lexical/semantic 的具体压缩结果无意义，只断言意图与路由开关。
        cases.append({
            "id": spec["id"], "type": "negative_semantic", "query": spec["query"],
            "expected_plan": {"intent": "semantic", "enable_dense": True, "enable_rerank": True},
            "expect_empty": True, "relevance": {}, "forbidden_ids": [],
            "evaluation": _evaluation_block(),
        })
    return cases


def _pick(fam: dict[str, list[str]], family: str, n: int, rng_offset: int = 0) -> list[str]:
    """从某族取前 n 个 id（确定性，按生成顺序）。"""
    ids = fam.get(family, [])
    return ids[rng_offset:rng_offset + n]


def _build_exact_lookup_cases(fam, by_id) -> list[dict]:
    """编号 / 邮箱 / 手机号精确定位：唯一命中，期望 intent=lookup。"""
    picks = (
        _pick(fam, "ml_llm", 2) + _pick(fam, "security_research", 2)
        + _pick(fam, "backend_go", 2) + _pick(fam, "blue_team", 2)
    )  # 8 个不同候选人
    cases = []
    # 候选人编号 ×4（第 2 条用小写测大小写容错）
    for i, rid in enumerate(picks[:4]):
        q = rid.lower() if i == 1 else rid
        cases.append({
            "id": f"exact_no_{rid.lower()}", "type": "exact_lookup", "query": q,
            "expected_plan": {"intent": "lookup", "lexical_query": rid, "semantic_query": rid,
                              "enable_dense": False, "enable_rerank": False},
            "relevance": {rid: 3}, "forbidden_ids": [], "evaluation": _evaluation_block(),
        })
    # 邮箱 ×2
    for rid in picks[4:6]:
        email = by_id[rid]["candidate"]["email"]
        cases.append({
            "id": f"exact_email_{rid.lower()}", "type": "exact_lookup", "query": email,
            "expected_plan": {"intent": "lookup", "lexical_query": email, "semantic_query": email,
                              "enable_dense": False, "enable_rerank": False},
            "relevance": {rid: 3}, "forbidden_ids": [], "evaluation": _evaluation_block(),
        })
    # 手机号 ×2
    for rid in picks[6:8]:
        phone = by_id[rid]["candidate"]["phone"]
        cases.append({
            "id": f"exact_phone_{rid.lower()}", "type": "exact_lookup", "query": phone,
            "expected_plan": {"intent": "lookup", "lexical_query": phone, "semantic_query": phone,
                              "enable_dense": False, "enable_rerank": False},
            "relevance": {rid: 3}, "forbidden_ids": [], "evaluation": _evaluation_block(),
        })
    return cases


def _build_entity_exact_cases(fam, by_id) -> list[dict]:
    """学校实体精确：相关集 = 库里 candidate.school 命中该校的人（动态圈定）。

    实体查询的相关集是"库里所有匹配该实体的人"，用 relevant_es_query 在当前索引
    上实时圈定，避免硬编码遗漏。期望 intent=keyword、dense=false。
    注意：主索引只有 candidate.school（最高学历院校），用 .keyword 精确匹配。
    """
    # 选库中确实高频出现的学校，保证相关集非空
    schools = ["北京邮电大学", "同济大学", "哈尔滨工业大学", "武汉大学",
               "南京大学", "东南大学", "山东大学", "上海交通大学"]
    cases = []
    for school in schools:
        cases.append({
            "id": f"entity_school_{school}", "type": "entity_exact", "query": school,
            "expected_plan": {"intent": "keyword", "lexical_query": school, "semantic_query": "",
                              "enable_dense": False, "enable_rerank": False},
            "relevant_es_query": {"term": {"candidate.school.keyword": school}},
            "forbidden_ids": [], "evaluation": _evaluation_block(),
        })
    return cases


def _build_major_query_cases(fam, by_id) -> list[dict]:
    """专业名查询：相关集为库里该专业的人（动态圈定），期望 keyword。"""
    majors = ["计算机科学与技术", "软件工程", "信息安全", "网络空间安全",
              "通信工程", "计算机技术", "网络工程", "信息管理与信息系统"]
    cases = []
    for major in majors:
        cases.append({
            "id": f"major_{major}", "type": "major_query", "query": major,
            "expected_plan": {"intent": "keyword", "lexical_query": major, "semantic_query": "",
                              "enable_dense": False, "enable_rerank": False},
            "relevant_es_query": {"term": {"candidate.major.keyword": major}},
            "forbidden_ids": [], "evaluation": _evaluation_block(),
        })
    return cases


def _build_structured_filter_cases(fam, by_id) -> list[dict]:
    """结构化约束 + 检索文本：技能词 + 学历/城市硬过滤。

    相关集直接由 ground-truth 派生 = 同族 ∩ 该学历 ∩ 期望城市含该市。组合从真实
    数据里挑选保证非空（≥2 人）。期望 intent=semantic。
    """
    # (id, query 文本中的技能片段, 族, 学历, 城市) —— 组合经数据验证 ≥2 人
    table = [
        ("sf_ml_wuhan_master", "RAG 向量检索 大模型", "ml_llm", "硕士", "武汉"),
        ("sf_go_chengdu_master", "Golang Kubernetes gRPC", "backend_go", "硕士", "成都"),
        ("sf_cpp_guangzhou_phd", "C++ RDMA 低延迟", "backend_cpp", "博士", "广州"),
        ("sf_blue_beijing_master", "蓝队 应急响应 威胁狩猎", "blue_team", "硕士", "北京"),
        ("sf_java_chengdu_master", "Java 架构 分布式事务", "backend_java", "硕士", "成都"),
        ("sf_redteam_shanghai_master", "内网渗透 横向移动", "red_team", "硕士", "上海"),
        ("sf_data_shenzhen_master", "数据分析 A/B 实验 归因", "data_analysis", "硕士", "深圳"),
        ("sf_frontend_suzhou_master", "前端 WebGL 可视化", "frontend", "硕士", "苏州"),
    ]
    out = []
    for cid, skill_text, family, degree, city in table:
        query = f"{city} {degree} {skill_text}"
        relevant = {
            rid: 3
            for rid in fam.get(family, [])
            if by_id[rid]["candidate"]["highest_degree"] == degree
            and city in by_id[rid]["application"]["expected_work_cities"]
        }
        assert len(relevant) >= 2, f"{cid} structured_filter 相关集不足: {len(relevant)}"
        out.append({
            "id": cid, "type": "structured_filter", "query": query,
            "expected_plan": {"intent": "semantic", "lexical_query": skill_text, "semantic_query": skill_text,
                              "enable_dense": True, "enable_rerank": True},
            "relevance": relevant, "forbidden_ids": [], "evaluation": _evaluation_block(),
        })
    return out


def _build_cross_language_cases(fam, by_id) -> list[dict]:
    """中英混合/英文 query：与 semantic 同主题，但用英文/缩写表达。

    元组第 5 项为 expected_lexical（期望压缩保留的核心词，子集判定），避免英文
    冠词/介词（a/with/and）等被要求逐字复现。
    """
    specs = [
        ("cross_rag", "Need a RAG engineer with LangChain, vector retrieval and LLM experience", ["ml_llm"], ["hn_search_ops"], "RAG LangChain LLM"),
        ("cross_go_micro", "Go backend, Kubernetes, gRPC, high concurrency microservices", ["backend_go"], [], "Go Kubernetes gRPC"),
        ("cross_blue_siem", "Blue team incident response with SIEM/ELK, threat hunting and ATT&CK", ["blue_team"], ["hn_it_ops"], "SIEM threat hunting ATT&CK"),
        ("cross_cpp_hft", "Low latency C++ HFT RDMA lock-free Linux", ["backend_cpp"], ["hn_cpp_biz"], "C++ RDMA"),
        ("cross_abtest", "A/B testing SQL Python retention and causal analysis", ["data_analysis"], ["hn_bi_report"], "SQL Python retention"),
        ("cross_devsecops", "DevSecOps SAST IAST SCA CI/CD secure SDLC", ["devsecops"], [], "DevSecOps SAST IAST"),
        ("cross_webgl", "WebGL Three.js shader digital twin frontend visualization", ["frontend"], [], "WebGL shader"),
        ("cross_pentest", "Red team penetration testing, AD domain, lateral movement, C2", ["red_team"], [], "penetration lateral movement C2"),
    ]
    cases = []
    for cid, query, core, forbidden_fams, expected_lexical in specs:
        relevance, forbidden = _grade_map({"core": core, "forbidden_families": forbidden_fams}, fam)
        cases.append({
            "id": cid, "type": "cross_language", "query": query,
            "expected_plan": _semantic_plan(query, expected_lexical),
            "relevance": relevance, "forbidden_ids": forbidden,
            "evaluation": _evaluation_block(),
        })
    return cases


BUILDERS = [
    _build_exact_lookup_cases,
    _build_entity_exact_cases,
    _build_major_query_cases,
    _build_skill_combo_cases,
    _build_structured_filter_cases,
    _build_semantic_cases,
    _build_cross_language_cases,
    _build_negative_cases,
    _build_jd_cases,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="从 ground-truth 派生 eval_queries.jsonl")
    parser.add_argument("-o", "--output", default="eval_queries.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fam, by_id = _load_ground_truth()
    all_cases: list[dict] = []
    for builder in BUILDERS:
        all_cases.extend(builder(fam, by_id))

    # 统计与一致性自检
    from collections import Counter
    type_counts = Counter(c["type"] for c in all_cases)
    ids = [c["id"] for c in all_cases]
    assert len(ids) == len(set(ids)), "case id 重复"
    for c in all_cases:
        rel = set(c.get("relevance") or {})
        forb = set(c.get("forbidden_ids") or [])
        overlap = rel & forb
        assert not overlap, f"{c['id']} relevance 与 forbidden 重叠: {overlap}"
        if c.get("expect_empty"):
            assert not rel, f"{c['id']} expect_empty 但有相关 id"

    print(f"生成 {len(all_cases)} 条 query")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:22} {n}")
    # 静态 relevance 的 query 平均相关数
    static = [c for c in all_cases if c.get("relevance") and not c.get("expect_empty")]
    if static:
        avg = sum(len(c["relevance"]) for c in static) / len(static)
        print(f"静态 relevance query 平均相关数: {avg:.1f}")
    dyn = [c for c in all_cases if "relevant_es_query" in c]
    print(f"动态 es_query 圈定相关集的 query: {len(dyn)}")

    if args.dry_run:
        print("(dry-run，未写文件)")
        return
    out = Path(args.output)
    with out.open("w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
