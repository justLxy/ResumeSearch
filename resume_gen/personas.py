"""岗位族画像与职业路径构造（确定性、纯 Python，不联网）。

一个"族"(family) 描述一条真实的职业方向：投递岗位、专业、研究方向、技能栈、
典型项目主题、实习岗位。族的技能重叠关系被检索评测用来派生弱相关分级，所以族 id
保持稳定（ml_llm / blue_team / ... / hn_*）。

本模块只产出结构化骨架（学历、院校段、技能、日期等）——这些字段确定性、可复现，
是评测 ground-truth 的来源。项目/实习的自由文本由 narrative 模块负责。
"""
from __future__ import annotations

import random
from datetime import date
from typing import Any

from resume_gen import reference as ref

# --- 学历与论文 ------------------------------------------------------------
DEGREE_LABELS = {
    "本科": ("本科", "学士"),
    "硕士": ("硕士研究生", "硕士"),
    "博士": ("博士研究生", "博士"),
}
# 各族默认学历权重（本科, 硕士, 博士）。研究/算法/低延迟偏高学历，测试/产品偏本科。
DEFAULT_DEGREE_WEIGHTS = (30, 55, 15)
COLLEGES = [
    "计算机学院", "计算机科学与技术学院", "软件学院", "信息科学与工程学院",
    "网络空间安全学院", "电子工程学院", "信息与通信工程学院", "人工智能学院",
    "数据科学与工程学院", "自动化学院",
]
PAPER_LEVELS_PG = ["无", "无", "EI", "SCI", "核心期刊", "CCF-B", "CCF-A"]


# --- 岗位族画像 ------------------------------------------------------------
# 每个族提供：投递岗位/编号、专业、研究方向、实验室、技能池、核心技能、
# project_themes（项目主题锚点，供 narrative 展开）、internship_roles（部门/职位锚点）、
# award_themes（竞赛主题）。domain_label 是给 LLM/回退文本的方向说明。
FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "ml_llm": {
        "weight": 20,
        "domain_label": "机器学习 / 大模型算法",
        "degree_weights": (18, 60, 22),
        "positions": ["机器学习工程师", "算法工程师", "机器学习工程师(LLM方向)", "NLP算法工程师", "大模型应用工程师"],
        "position_codes": ["A0009", "A0010", "A0011"],
        "majors": ["计算机技术", "计算机科学与技术", "人工智能", "模式识别与智能系统", "软件工程", "电子信息"],
        "research_directions": ["自然语言处理与推荐系统", "大语言模型与检索增强", "知识图谱与问答", "深度学习与表示学习", "多模态理解"],
        "labs": ["智能信息处理实验室", "自然语言处理实验室", "机器学习重点实验室", "认知计算实验室"],
        "skills_pool": ["Python", "PyTorch", "TensorFlow", "NLP", "RAG", "LangChain", "向量检索", "大模型微调", "MySQL", "Spark", "Milvus", "Transformer"],
        "core_skills": ["Python", "PyTorch"],
        "project_themes": [
            "企业知识库检索增强问答系统", "信息流推荐召回与精排优化", "多模态内容理解平台",
            "对话式智能客服机器人", "文本分类与命名实体识别", "大模型指令微调与评测",
            "搜索语义相关性建模", "智能导诊与意图识别系统",
        ],
        "internship_roles": [("算法部", "算法实习生"), ("机器学习平台部", "机器学习实习生"), ("搜索推荐部", "算法工程师(实习)"), ("大模型团队", "NLP实习生")],
        "award_themes": [("中国高校计算机大赛人工智能创意赛", ["一等奖", "二等奖"]), ("研究生数学建模竞赛", ["一等奖", "二等奖"]), ("Kaggle 竞赛", ["Top 5%", "银牌"])],
    },
    "security_research": {
        "weight": 12,
        "domain_label": "安全研究 / 漏洞挖掘",
        "degree_weights": (22, 55, 23),
        "positions": ["安全研究员", "安全研究员(漏洞挖掘)", "高级安全研究员", "安全研究员(逆向分析)"],
        "position_codes": ["S0101", "S0102", "S0103"],
        "majors": ["网络空间安全", "信息安全", "计算机科学与技术", "软件工程"],
        "research_directions": ["二进制漏洞挖掘", "模糊测试与符号执行", "内核安全", "软件供应链安全", "逆向工程"],
        "labs": ["系统安全实验室", "网络攻防实验室", "可信计算重点实验室"],
        "skills_pool": ["C", "C++", "Python", "IDA Pro", "Ghidra", "GDB", "汇编", "Linux", "WinDbg", "AFL", "符号执行"],
        "core_skills": ["C++", "IDA Pro"],
        "project_themes": [
            "浏览器引擎模糊测试与漏洞挖掘", "内核提权漏洞研究", "软件供应链组件审计",
            "固件逆向与漏洞分析", "反序列化与内存破坏漏洞研究", "自动化漏洞挖掘平台",
        ],
        "internship_roles": [("安全研究部", "安全研究实习生"), ("漏洞研究团队", "二进制安全实习生"), ("高级威胁研究院", "逆向分析实习生")],
        "award_themes": [("全国大学生信息安全竞赛", ["一等奖", "二等奖"]), ("强网杯网络安全挑战赛", ["二等奖", "三等奖"]), ("DEF CON CTF", ["决赛圈"])],
    },
    "blue_team": {
        "weight": 12,
        "domain_label": "蓝队 / 安全运营与应急响应",
        "positions": ["蓝队应急响应工程师", "安全运营工程师(SOC)", "威胁狩猎工程师", "安全分析师"],
        "position_codes": ["S0201", "S0202", "S0203"],
        "majors": ["网络空间安全", "信息安全", "计算机科学与技术", "网络工程"],
        "research_directions": ["威胁检测与响应", "日志分析与取证", "ATT&CK威胁建模", "威胁情报分析"],
        "labs": ["安全运营实验室", "威胁情报实验室"],
        "skills_pool": ["Python", "Splunk", "ELK", "Suricata", "Wireshark", "SIEM", "Linux", "Sigma", "YARA", "ATT&CK", "SOAR"],
        "core_skills": ["Python", "SIEM"],
        "project_themes": [
            "企业安全事件应急响应平台", "威胁狩猎与日志取证系统", "SOC 告警降噪与自动化编排",
            "勒索软件溯源与处置", "内网横向移动检测", "威胁情报聚合与研判平台",
        ],
        "internship_roles": [("安全运营中心", "安全运营实习生"), ("应急响应团队", "应急响应实习生"), ("威胁情报中心", "威胁分析实习生")],
        "award_themes": [("网络安全应急响应技能大赛", ["二等奖", "三等奖"]), ("护网行动", ["优秀个人"])],
    },
    "red_team": {
        "weight": 12,
        "domain_label": "红队 / 渗透测试与攻防",
        "positions": ["红队攻防工程师", "渗透测试工程师", "高级渗透测试工程师", "红队开发工程师"],
        "position_codes": ["S0301", "S0302", "S0303"],
        "majors": ["网络空间安全", "信息安全", "计算机科学与技术"],
        "research_directions": ["内网渗透与横向移动", "Web安全", "免杀与对抗", "红队武器化开发"],
        "labs": ["攻防对抗实验室", "红队技术实验室"],
        "skills_pool": ["Python", "Go", "Cobalt Strike", "Burp Suite", "Metasploit", "C#", "PowerShell", "Linux", "内网渗透", "免杀"],
        "core_skills": ["Python", "内网渗透"],
        "project_themes": [
            "红队内网渗透演练平台", "Web 应用渗透测试", "AD 域横向移动与权限维持",
            "C2 隐蔽通信框架开发", "免杀与对抗研究", "自动化资产测绘与漏洞利用",
        ],
        "internship_roles": [("红队", "红队实习生"), ("渗透测试团队", "渗透测试实习生"), ("攻防实验室", "安全服务实习生")],
        "award_themes": [("强网杯网络安全挑战赛", ["二等奖", "三等奖"]), ("网鼎杯", ["三等奖"])],
    },
    "devsecops": {
        "weight": 10,
        "domain_label": "DevSecOps / 应用与云原生安全",
        "positions": ["DevSecOps工程师", "安全平台开发工程师", "应用安全工程师", "云安全工程师"],
        "position_codes": ["S0401", "S0402", "S0403"],
        "majors": ["计算机科学与技术", "软件工程", "网络空间安全"],
        "research_directions": ["软件供应链安全", "CI/CD安全", "云原生安全", "运行时应用自保护"],
        "labs": ["应用安全实验室", "DevSecOps实验室"],
        "skills_pool": ["Java", "Go", "Python", "SAST", "SCA", "IAST", "Docker", "Kubernetes", "Jenkins", "GitLab CI", "OWASP"],
        "core_skills": ["SAST", "Docker"],
        "project_themes": [
            "DevSecOps 安全左移平台", "容器与镜像安全扫描系统", "软件成分分析与许可证合规",
            "研发流水线安全卡点", "云原生运行时防护", "API 资产梳理与风险治理",
        ],
        "internship_roles": [("安全平台研发部", "安全研发实习生"), ("应用安全团队", "应用安全实习生")],
        "award_themes": [("全国大学生软件测试大赛", ["二等奖"]), ("百度之星程序设计大赛", ["三等奖"])],
    },
    "backend_go": {
        "weight": 12,
        "domain_label": "Go 后端 / 云原生基础设施",
        "positions": ["后端开发工程师(Go)", "Go云原生工程师", "高级后端工程师(Go)", "云原生基础设施工程师"],
        "position_codes": ["B0501", "B0502", "B0503"],
        "majors": ["计算机科学与技术", "软件工程", "通信工程"],
        "research_directions": ["分布式系统", "云原生架构", "高并发服务"],
        "labs": ["分布式系统实验室", "云计算实验室"],
        "skills_pool": ["Go", "Kubernetes", "gRPC", "Docker", "etcd", "Redis", "MySQL", "Prometheus", "微服务", "Kafka"],
        "core_skills": ["Go", "Kubernetes"],
        "project_themes": [
            "Kubernetes Operator 与服务网格", "高并发 API 网关", "分布式消息队列服务",
            "云原生可观测性平台", "容器调度与弹性伸缩", "多租户配置中心",
        ],
        "internship_roles": [("基础架构部", "后端开发实习生"), ("云原生团队", "Go开发实习生"), ("中间件团队", "服务端开发实习生")],
        "award_themes": [("ACM-ICPC 区域赛", ["铜奖", "银奖"]), ("字节跳动 Byte Camp", ["优秀学员"])],
    },
    "backend_java": {
        "weight": 12,
        "domain_label": "Java 后端 / 高并发业务系统",
        "positions": ["后端开发工程师(Java)", "Java架构师", "高级Java开发工程师", "资深Java开发工程师"],
        "position_codes": ["B0601", "B0602", "B0603"],
        "majors": ["计算机科学与技术", "软件工程", "信息管理与信息系统"],
        "research_directions": ["分布式事务", "高可用架构", "中间件"],
        "labs": ["软件工程实验室", "分布式系统实验室"],
        "skills_pool": ["Java", "Spring Boot", "Spring Cloud", "MySQL", "Redis", "RocketMQ", "Kafka", "JVM", "分库分表", "Dubbo"],
        "core_skills": ["Java", "Spring Boot"],
        "project_themes": [
            "金融级高并发交易系统", "电商订单与库存中心", "企业级微服务中台",
            "支付清结算系统", "会员权益与营销平台", "分布式任务调度系统",
        ],
        "internship_roles": [("交易研发部", "Java开发实习生"), ("电商技术部", "后端开发实习生"), ("支付团队", "服务端实习生")],
        "award_themes": [("蓝桥杯软件大赛", ["二等奖", "三等奖"]), ("中国大学生服务外包创新创业大赛", ["二等奖"])],
    },
    "backend_cpp": {
        "weight": 8,
        "domain_label": "C++ 后端 / 低延迟高性能系统",
        "degree_weights": (25, 52, 23),
        "positions": ["后端开发工程师(C++)", "C++低延迟工程师", "高级C++研发工程师", "C++系统工程师"],
        "position_codes": ["B0701", "B0702", "B0703"],
        "majors": ["计算机科学与技术", "电子与通信工程", "软件工程"],
        "research_directions": ["低延迟系统", "高性能计算", "网络编程"],
        "labs": ["高性能计算实验室", "网络与系统实验室"],
        "skills_pool": ["C++", "C", "RDMA", "无锁编程", "Linux", "DPDK", "Redis", "TCP/IP", "性能调优", "协程"],
        "core_skills": ["C++", "无锁编程"],
        "project_themes": [
            "超低延迟交易基础设施", "高性能 KV 存储引擎", "实时行情接入网关",
            "内核旁路网络收发框架", "高频风控计算引擎", "分布式缓存内核优化",
        ],
        "internship_roles": [("核心系统部", "C++开发实习生"), ("高性能计算团队", "系统开发实习生")],
        "award_themes": [("ACM-ICPC 区域赛", ["银奖", "金奖"]), ("CCF CCSP", ["二等奖"])],
    },
    "frontend": {
        "weight": 10,
        "domain_label": "前端 / Web 与可视化",
        "positions": ["前端开发工程师", "高级前端开发工程师", "前端可视化工程师", "Web前端工程师"],
        "position_codes": ["F0801", "F0802", "F0803"],
        "majors": ["计算机科学与技术", "软件工程", "数字媒体技术"],
        "research_directions": ["前端工程化", "数据可视化", "WebGL渲染"],
        "labs": ["人机交互实验室", "可视化实验室"],
        "skills_pool": ["JavaScript", "TypeScript", "Vue", "React", "WebGL", "Three.js", "Vite", "CSS", "Node.js", "ECharts"],
        "core_skills": ["JavaScript", "TypeScript"],
        "project_themes": [
            "大屏数据可视化平台", "企业级组件库与微前端", "三维地图可视化引擎",
            "低代码运营后台", "实时协同编辑器", "前端性能监控体系",
        ],
        "internship_roles": [("前端研发部", "前端开发实习生"), ("可视化团队", "前端工程师(实习)")],
        "award_themes": [("中国大学生计算机设计大赛", ["二等奖", "三等奖"]), ("GeekPwn 极客大赛", ["优胜奖"])],
    },
    "data_analysis": {
        "weight": 10,
        "domain_label": "数据分析 / 增长",
        "positions": ["数据分析师", "数据分析师(增长)", "高级数据分析师", "商业分析师"],
        "position_codes": ["D0901", "D0902", "D0903"],
        "majors": ["统计学", "应用统计", "数据科学与大数据技术", "计算机科学与技术", "经济学"],
        "research_directions": ["因果推断与实验设计", "用户增长", "指标体系建设"],
        "labs": ["数据科学实验室", "统计与计算实验室"],
        "skills_pool": ["SQL", "Python", "A/B测试", "数据可视化", "Tableau", "Hive", "Spark", "因果推断", "指标体系", "Pandas"],
        "core_skills": ["SQL", "A/B测试"],
        "project_themes": [
            "用户增长 A/B 实验平台", "经营分析指标看板", "用户留存与流失归因",
            "营销投放归因建模", "转化漏斗与异动分析", "用户分层与画像体系",
        ],
        "internship_roles": [("数据分析部", "数据分析实习生"), ("增长团队", "商业分析实习生"), ("战略分析部", "数据分析师(实习)")],
        "award_themes": [("全国大学生统计建模大赛", ["二等奖", "三等奖"]), ("美国大学生数学建模竞赛", ["Meritorious"])],
    },
    "testing": {
        "weight": 8,
        "domain_label": "测试 / 质量保障",
        "degree_weights": (45, 48, 7),
        "positions": ["测试工程师", "自动化测试工程师", "测试开发工程师", "性能测试工程师"],
        "position_codes": ["A0014", "A0015", "A0016"],
        "majors": ["电子与通信工程", "计算机科学与技术", "软件工程", "通信工程"],
        "research_directions": ["自动化测试", "性能测试", "测试平台建设"],
        "labs": ["软件质量实验室", "嵌入式系统实验室"],
        "skills_pool": ["Python", "C", "C++", "Selenium", "JMeter", "pytest", "MySQL", "Linux", "Appium", "Jenkins"],
        "core_skills": ["Python", "自动化测试"],
        "project_themes": [
            "接口与 UI 自动化测试平台", "核心链路性能压测与调优", "持续集成质量门禁",
            "移动端兼容性测试体系", "测试用例管理与覆盖率统计", "混沌工程与稳定性演练",
        ],
        "internship_roles": [("测试部", "测试开发实习生"), ("质量工程部", "自动化测试实习生")],
        "award_themes": [("全国大学生软件测试大赛", ["二等奖", "三等奖"])],
    },
    "product": {
        "weight": 6,
        "domain_label": "产品 / 项目管理",
        "degree_weights": (40, 55, 5),
        "positions": ["安全产品经理", "技术项目经理(TPM)", "产品经理", "产品经理(数据方向)"],
        "position_codes": ["P1001", "P1002", "P1003"],
        "majors": ["信息管理与信息系统", "软件工程", "计算机科学与技术", "工商管理"],
        "research_directions": ["安全产品设计", "项目管理", "数据产品"],
        "labs": ["信息系统实验室"],
        "skills_pool": ["需求分析", "Axure", "SQL", "项目管理", "SASE", "ZTNA", "零信任", "数据产品", "Visio", "竞品分析"],
        "core_skills": ["需求分析", "项目管理"],
        "project_themes": [
            "零信任访问产品规划", "安全运营平台项目管理", "数据分析产品看板设计",
            "SASE 产品需求与落地", "企业协同工具改版", "B 端工作台体验优化",
        ],
        "internship_roles": [("产品部", "产品实习生"), ("项目管理办公室", "项目管理实习生")],
        "award_themes": [("全国大学生创新创业大赛", ["二等奖", "三等奖"])],
    },
}


# --- Hard negative 族：与热门检索主题"沾边但不对"，用于考验精排与 forbidden@10 ---
HARD_NEGATIVE_PROFILES: dict[str, dict[str, Any]] = {
    "hn_search_ops": {  # 像 RAG，实为传统搜索/ES 运维
        "weight": 6,
        "domain_label": "搜索/ES 运维（非算法）",
        "degree_weights": (55, 42, 3),
        "positions": ["搜索运维工程师", "ES运维工程师", "搜索平台运维"],
        "position_codes": ["H0101"],
        "majors": ["计算机科学与技术", "软件工程", "网络工程"],
        "research_directions": ["搜索引擎运维", "日志检索平台"],
        "labs": ["信息检索实验室"],
        "skills_pool": ["Elasticsearch", "Lucene", "Java", "Linux", "Logstash", "Kibana", "检索", "分片调优"],
        "core_skills": ["Elasticsearch", "检索"],
        "project_themes": ["企业搜索与日志检索平台运维", "ES 集群容量与索引治理"],
        "internship_roles": [("运维部", "搜索运维实习生")],
        "award_themes": [],
        "negative_note": "只做检索集群/索引运维，不涉及大模型、向量语义召回与 RAG 生成",
    },
    "hn_it_ops": {  # 像蓝队，实为普通运维/网管
        "weight": 6,
        "domain_label": "IT/网络运维（非安全分析）",
        "degree_weights": (60, 38, 2),
        "positions": ["IT运维工程师", "网络运维工程师", "系统运维工程师"],
        "position_codes": ["H0201"],
        "majors": ["网络工程", "计算机科学与技术", "通信工程"],
        "research_directions": ["IT基础设施运维", "网络监控"],
        "labs": ["网络工程实验室"],
        "skills_pool": ["Linux", "Shell", "Zabbix", "日志", "监控", "网络运维", "MySQL", "Nginx"],
        "core_skills": ["Linux", "监控"],
        "project_themes": ["企业 IT 基础设施监控运维", "机房网络与服务器巡检"],
        "internship_roles": [("运维部", "运维实习生")],
        "award_themes": [],
        "negative_note": "只做日常运维与监控值守，不涉及威胁狩猎、攻击链取证与安全分析",
    },
    "hn_cpp_biz": {  # 像低延迟 C++，实为普通 C++ 业务开发
        "weight": 6,
        "domain_label": "C++ 桌面/业务开发（非低延迟）",
        "degree_weights": (55, 43, 2),
        "positions": ["C++业务开发工程师", "桌面应用开发工程师(C++)", "C++软件工程师"],
        "position_codes": ["H0301"],
        "majors": ["计算机科学与技术", "软件工程"],
        "research_directions": ["桌面应用开发", "业务系统开发"],
        "labs": ["软件工程实验室"],
        "skills_pool": ["C++", "Qt", "MySQL", "Windows", "MFC", "业务开发"],
        "core_skills": ["C++", "Qt"],
        "project_themes": ["桌面客户端业务系统", "企业管理软件表单模块"],
        "internship_roles": [("研发部", "C++开发实习生")],
        "award_themes": [],
        "negative_note": "只做桌面/业务功能开发，不涉及低延迟、RDMA、无锁编程等高性能场景",
    },
    "hn_bi_report": {  # 像数据分析，实为 BI 报表开发
        "weight": 6,
        "domain_label": "BI 报表/数仓开发（非增长分析）",
        "degree_weights": (55, 43, 2),
        "positions": ["BI报表开发工程师", "报表开发工程师", "数仓报表开发"],
        "position_codes": ["H0401"],
        "majors": ["信息管理与信息系统", "计算机科学与技术", "统计学"],
        "research_directions": ["报表开发", "数据仓库"],
        "labs": ["信息系统实验室"],
        "skills_pool": ["SQL", "Tableau", "PowerBI", "ETL", "Excel", "数据仓库", "Kettle"],
        "core_skills": ["SQL", "Tableau"],
        "project_themes": ["经营报表与数据仓库开发", "固定报表与数据集市建设"],
        "internship_roles": [("数据部", "报表开发实习生")],
        "award_themes": [],
        "negative_note": "只做固定报表与 ETL 开发，不涉及 A/B 实验、因果推断与增长策略分析",
    },
}

ALL_PROFILES = {**FAMILY_PROFILES, **HARD_NEGATIVE_PROFILES}


# --- 职业路径构造 ----------------------------------------------------------
def choose_degree(rng: random.Random, profile: dict[str, Any]) -> str:
    weights = profile.get("degree_weights", DEFAULT_DEGREE_WEIGHTS)
    return rng.choices(["本科", "硕士", "博士"], weights=weights, k=1)[0]


def choose_grad_year(rng: random.Random, current_year: int) -> int:
    """毕业年份：以当前招聘季为中心，覆盖应届/往届/在读。"""
    offsets = [-2, -1, 0, 0, 1, 1]  # 略偏向应届与在读
    return current_year + rng.choice(offsets)


def build_education(
    rng: random.Random,
    profile: dict[str, Any],
    degree: str,
    grad_year: int,
    apply_year: int,
) -> list[dict[str, Any]]:
    """构造教育经历：本科一段；硕士/博士再加一段（含研究方向/实验室/论文等级）。

    院校用真实名单采样，本科与研究生可换校。研究方向只挂在研究生段（本科为"无"），
    与两份真实简历一致。
    """
    ug_school, pg_school = ref.sample_school_progression(rng)
    edu: list[dict[str, Any]] = []

    ug_len = 4
    ug_start_year = grad_year - (ug_len if degree == "本科" else (ug_len + 3 if degree == "硕士" else ug_len + 6))
    ug_start = date(ug_start_year, 9, 1)
    ug_end = date(ug_start_year + ug_len, 6, rng.randint(18, 30))
    ug_major = rng.choice(profile["majors"])
    edu.append({
        "start_date": ug_start.isoformat(),
        "end_date": ug_end.isoformat(),
        "school": ug_school,
        "college": rng.choice(COLLEGES),
        "major": ug_major,
        "education_level": "本科",
        "degree": "学士",
        "research_direction": "无",
        "lab_name": None,
        "paper_level": "无",
        "is_current": False,
    })

    if degree in ("硕士", "博士"):
        pg_len = 3 if degree == "硕士" else 4
        pg_start = date(ug_end.year, 9, 1)
        # 在读判定：毕业年份还没到招聘年，或就是当年但尚未毕业。
        is_current = grad_year >= apply_year
        pg_end = None if is_current else date(grad_year, 6, rng.randint(18, 30))
        level_label, deg_label = DEGREE_LABELS[degree]
        # 研究生专业多与本科相关，但允许跨专业升学。
        pg_major = ug_major if rng.random() < 0.6 else rng.choice(profile["majors"])
        edu.append({
            "start_date": pg_start.isoformat(),
            "end_date": pg_end.isoformat() if pg_end else None,
            "school": pg_school,
            "college": rng.choice(COLLEGES),
            "major": pg_major,
            "education_level": level_label,
            "degree": deg_label,
            "research_direction": rng.choice(profile["research_directions"]),
            "lab_name": rng.choice(profile["labs"]),
            "paper_level": rng.choice(PAPER_LEVELS_PG),
            "is_current": is_current,
        })
    return edu


def build_birth_date(rng: random.Random, degree: str, grad_year: int) -> str:
    """出生日期：由学历与毕业年份反推大致年龄，加抖动。"""
    typical_grad_age = {"本科": 22, "硕士": 25, "博士": 29}[degree]
    birth_year = grad_year - typical_grad_age - rng.randint(0, 2)
    return f"{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def build_skills(rng: random.Random, profile: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """技能标签 + it_skill_items（含熟练度/时长/主次编程语言，与真实简历一致）。"""
    pool = profile["skills_pool"]
    core = profile["core_skills"]
    n = rng.randint(4, min(8, len(pool)))
    chosen = list(dict.fromkeys(core + rng.sample(pool, k=min(n, len(pool)))))
    rng.shuffle(chosen)

    programming = ("Python", "Java", "Go", "C", "C++", "JavaScript", "TypeScript", "C#")
    langs = [s for s in chosen if s in programming]
    primary = langs[0] if langs else (chosen[0] if chosen else "Python")
    others = "、".join(langs[1:3]) if len(langs) > 1 else None

    # 主技能条目 1~2 条（真实简历里 IT技能通常 1 条，偶有多条）。
    items = [{
        "skill_name": primary,
        "duration": rng.choice(["1年以下", "1年", "2年", "3年", "3年以上"]),
        "proficiency": rng.choice(["了解", "一般", "熟练", "精通"]),
        "primary_languages": primary,
        "other_languages": others,
    }]
    if len(chosen) > 4 and rng.random() < 0.3:
        second = chosen[1] if chosen[1] != primary else (chosen[2] if len(chosen) > 2 else primary)
        items.append({
            "skill_name": second,
            "duration": rng.choice(["1年以下", "1年", "2年"]),
            "proficiency": rng.choice(["了解", "一般", "熟练"]),
            "primary_languages": primary,
            "other_languages": others,
        })
    return chosen, items
