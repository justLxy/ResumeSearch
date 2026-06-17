from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import requests

from import_to_es import (
    DEFAULT_ALIAS,
    DEFAULT_ES_URL,
    DEFAULT_INDEX,
    INDEX_BODY,
    _bulk_index,
    _enrich_doc,
    _request,
    _switch_alias,
    _versioned_index_name,
    add_doc_embeddings,
)


SURNAMES = [
    "赵",
    "钱",
    "孙",
    "李",
    "周",
    "吴",
    "郑",
    "王",
    "冯",
    "陈",
    "褚",
    "卫",
    "蒋",
    "沈",
    "韩",
    "杨",
    "朱",
    "秦",
    "尤",
    "许",
    "何",
    "吕",
    "施",
    "张",
    "孔",
    "曹",
    "严",
    "华",
    "金",
    "魏",
]
GIVEN_NAMES = [
    "子轩",
    "雨桐",
    "浩然",
    "梓涵",
    "一诺",
    "俊杰",
    "思源",
    "嘉怡",
    "晨曦",
    "睿哲",
    "欣然",
    "博文",
    "明轩",
    "可欣",
    "宇航",
    "佳宁",
    "昊天",
    "诗涵",
    "景行",
    "语嫣",
    "泽宇",
    "若曦",
    "承宇",
    "书瑶",
    "子墨",
    "晓彤",
    "辰逸",
    "静怡",
    "鸿煊",
    "雅婷",
]
SCHOOLS = [
    ("北京邮电大学", "计算机学院"),
    ("北京理工大学", "软件学院"),
    ("哈尔滨工业大学", "计算学部"),
    ("西安电子科技大学", "网络与信息安全学院"),
    ("电子科技大学", "信息与软件工程学院"),
    ("南京大学", "软件学院"),
    ("武汉大学", "计算机学院"),
    ("山东大学", "网络空间安全学院"),
    ("华中科技大学", "人工智能与自动化学院"),
    ("北京交通大学", "计算机与信息技术学院"),
]
MAJORS = [
    "计算机科学与技术",
    "软件工程",
    "网络空间安全",
    "信息安全",
    "数据科学与大数据技术",
    "人工智能",
    "电子信息工程",
    "计算机技术",
]
CITIES = ["北京", "上海", "深圳", "广州", "杭州", "南京", "成都", "武汉", "西安", "济南"]
POSITIONS = [
    ("A0009", "机器学习工程师"),
    ("A0010", "后端开发工程师"),
    ("A0011", "安全研究员"),
    ("A0012", "数据分析师"),
    ("A0013", "前端开发工程师"),
    ("A0014", "测试工程师"),
    ("A0015", "运维开发工程师"),
    ("A0016", "算法工程师"),
    ("A0017", "Java开发工程师"),
    ("A0018", "安全运营工程师"),
]
SKILL_POOLS = {
    "机器学习工程师": ["Python", "PyTorch", "TensorFlow", "机器学习", "NLP", "推荐系统", "SQL"],
    "算法工程师": ["Python", "C++", "深度学习", "图像识别", "NLP", "数据结构", "Linux"],
    "后端开发工程师": ["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "Docker", "Linux"],
    "Java开发工程师": ["Java", "Spring Cloud", "MySQL", "Redis", "微服务", "JVM", "Linux"],
    "前端开发工程师": ["JavaScript", "TypeScript", "Vue", "React", "HTML", "CSS", "Webpack"],
    "测试工程师": ["Python", "Selenium", "Pytest", "接口测试", "性能测试", "Linux", "SQL"],
    "运维开发工程师": ["Python", "Go", "Kubernetes", "Docker", "Prometheus", "Linux", "Shell"],
    "安全研究员": ["漏洞挖掘", "逆向分析", "Python", "C", "Web安全", "二进制安全", "Linux"],
    "安全运营工程师": ["SIEM", "应急响应", "日志分析", "Python", "威胁情报", "Linux", "SQL"],
    "数据分析师": ["Python", "SQL", "Pandas", "Spark", "数据仓库", "Tableau", "统计分析"],
}
COMPANIES = [
    "奇安信科技集团",
    "百度在线网络技术",
    "腾讯科技",
    "阿里云计算",
    "字节跳动",
    "京东科技",
    "美团",
    "小米科技",
    "网易杭州研究院",
    "滴滴出行",
]
POSITION_BACKGROUNDS = {
    "机器学习工程师": {
        "departments": ["智能安全实验室", "推荐算法组", "NLP平台组", "风控建模组"],
        "research": ["自然语言处理", "异常检测", "推荐召回", "知识图谱", "模型压缩"],
        "labs": ["智能安全实验室", "数据智能实验室", "认知计算实验室"],
        "projects": [
            ("安全告警语义聚类系统", "对告警标题和处置记录做向量化聚类，识别重复告警和相似攻击路径。"),
            ("简历技能画像抽取模型", "从教育、项目和实习文本中抽取技能实体，并生成候选人能力标签。"),
            ("弱监督风险样本挖掘平台", "使用规则标注和主动学习扩充训练集，降低人工标注成本。"),
            ("日志异常检测模型服务", "基于时序特征和自编码器发现主机异常行为。"),
            ("中文技术问答匹配模型", "训练双塔召回模型匹配问题、技能和项目经验。"),
            ("模型评测与漂移监控平台", "跟踪线上模型效果、样本分布和阈值稳定性。"),
        ],
        "internship_actions": [
            "清洗告警语料并训练文本分类模型",
            "搭建向量召回实验并对比 BM25、双塔和重排效果",
            "将离线模型封装为批处理推理服务",
            "设计特征统计报表并分析误召回样本",
        ],
        "outcomes": [
            "Top10 召回率提升 12%",
            "重复告警合并率提升 18%",
            "模型评估耗时减少 35%",
            "人工复核样本量下降 22%",
        ],
    },
    "后端开发工程师": {
        "departments": ["云平台研发部", "账号权限组", "交易中台组", "搜索服务组"],
        "research": ["分布式系统", "服务治理", "高并发架构", "缓存一致性"],
        "labs": ["软件工程实验室", "分布式系统实验室", "云计算实验室"],
        "projects": [
            ("权限中心灰度发布平台", "为多业务线提供角色、资源和策略管理能力。"),
            ("订单事件异步处理服务", "基于消息队列拆分订单状态流转和补偿任务。"),
            ("配置中心审计系统", "记录配置变更、审批和回滚链路，支持差异对比。"),
            ("高并发库存扣减服务", "使用 Redis 预扣和数据库最终校准保证库存一致性。"),
            ("搜索接口聚合网关", "统一封装候选人、岗位和部门检索接口。"),
            ("内部工单流转系统", "支持 SLA 计时、催办和跨部门协作。"),
        ],
        "internship_actions": [
            "拆分单体接口并补齐服务契约测试",
            "优化慢 SQL、缓存热点和接口超时问题",
            "接入 Kafka 消费链路并处理幂等写入",
            "实现后台管理接口和权限校验中间件",
        ],
        "outcomes": [
            "接口 P95 延迟下降 40%",
            "批处理吞吐提升 2 倍",
            "线上重复提交问题基本消除",
            "核心接口错误率下降 30%",
        ],
    },
    "安全研究员": {
        "departments": ["漏洞研究部", "攻防实验室", "二进制安全组", "Web安全组"],
        "research": ["Web 漏洞挖掘", "二进制分析", "协议逆向", "沙箱逃逸"],
        "labs": ["网络攻防实验室", "可信计算实验室", "系统安全实验室"],
        "projects": [
            ("Web 资产漏洞指纹库", "沉淀 CMS、框架和组件指纹，关联历史漏洞验证脚本。"),
            ("IoT 固件敏感接口扫描器", "解析固件文件系统并定位硬编码密钥和危险接口。"),
            ("二进制崩溃样本分析平台", "自动化收集 crash、去重堆栈并生成初步分析报告。"),
            ("供应链组件风险识别工具", "分析依赖版本、CVE 和补丁状态并生成风险分级。"),
            ("红队入口资产画像系统", "聚合端口、证书、域名和指纹信息辅助攻击面分析。"),
            ("PoC 自动验证沙箱", "隔离执行验证脚本并记录流量、日志和结果证据。"),
        ],
        "internship_actions": [
            "编写漏洞验证脚本并补充复现文档",
            "逆向分析协议字段和加密流程",
            "搭建批量指纹识别任务并清理误报",
            "参与攻防演练资产梳理和入口验证",
        ],
        "outcomes": [
            "沉淀 30+ 条有效漏洞指纹",
            "误报率下降 25%",
            "漏洞验证平均耗时缩短 50%",
            "发现并复现多个中高危风险点",
        ],
    },
    "数据分析师": {
        "departments": ["经营分析组", "数据平台部", "风控分析组", "增长分析组"],
        "research": ["数据挖掘", "统计建模", "用户画像", "指标体系"],
        "labs": ["大数据实验室", "数据科学实验室", "商业智能实验室"],
        "projects": [
            ("安全产品续费预测看板", "整合客户使用、工单和合同数据预测续费概率。"),
            ("招聘漏斗转化分析模型", "分析投递、筛选、面试和录用阶段的转化瓶颈。"),
            ("运营活动效果归因报表", "拆解渠道、地区和客户分层对转化率的影响。"),
            ("数据质量巡检规则平台", "对核心宽表做空值、波动和延迟监控。"),
            ("客户健康度评分体系", "基于登录、告警处置和工单响应构建客户分层。"),
            ("销售线索优先级模型", "融合线索来源、行业和历史转化数据生成跟进优先级。"),
        ],
        "internship_actions": [
            "搭建 BI 仪表盘并梳理指标口径",
            "编写 SQL 任务产出周报和异常归因",
            "用 Python 做分群分析和可视化",
            "清理维表映射并补充数据质量规则",
        ],
        "outcomes": [
            "核心报表出数时间缩短 60%",
            "定位 8 类高频数据口径问题",
            "业务复盘效率明显提升",
            "线索命中率提升 15%",
        ],
    },
    "前端开发工程师": {
        "departments": ["安全运营前端组", "数据可视化组", "低代码平台组", "体验工程组"],
        "research": ["前端工程化", "数据可视化", "交互设计", "组件体系"],
        "labs": ["人机交互实验室", "软件工程实验室", "可视化分析实验室"],
        "projects": [
            ("安全态势可视化大屏", "展示资产风险、告警趋势和攻击路径联动图。"),
            ("候选人检索工作台", "支持筛选、详情抽屉和高亮片段展示。"),
            ("低代码表单配置器", "通过 JSON Schema 生成动态表单和校验规则。"),
            ("组件库主题切换系统", "沉淀表格、筛选器、弹窗和图表主题规范。"),
            ("前端性能监控面板", "采集首屏、接口和资源加载指标并关联发布版本。"),
            ("告警处置流程编排页面", "用拖拽节点配置审批、通知和自动化动作。"),
        ],
        "internship_actions": [
            "实现复杂筛选器、虚拟列表和详情抽屉",
            "封装图表组件并处理大数据量渲染",
            "补齐组件单测和 Storybook 示例",
            "优化首屏资源拆包和接口并发加载",
        ],
        "outcomes": [
            "首屏时间下降 35%",
            "表格大数据滚动保持稳定",
            "组件复用覆盖 4 条业务线",
            "用户操作路径减少 3 步",
        ],
    },
    "测试工程师": {
        "departments": ["质量保障部", "自动化测试组", "性能测试组", "安全产品测试组"],
        "research": ["测试工程", "接口自动化", "性能压测", "质量度量"],
        "labs": ["软件质量实验室", "云测试实验室", "软件工程实验室"],
        "projects": [
            ("接口自动化回归平台", "管理接口用例、环境变量和断言规则。"),
            ("安全产品压测基线系统", "对告警写入、查询和报表生成链路做容量评估。"),
            ("缺陷聚类与质量看板", "按模块、版本和严重级别统计缺陷趋势。"),
            ("UI 冒烟测试流水线", "覆盖登录、检索、筛选和详情核心路径。"),
            ("测试数据构造工具", "快速生成候选人、岗位和告警模拟数据。"),
            ("稳定性巡检脚本平台", "定时执行环境检查和异常通知。"),
        ],
        "internship_actions": [
            "编写接口自动化用例并接入 CI",
            "设计压测场景并分析瓶颈指标",
            "维护测试环境和 mock 数据构造脚本",
            "复现线上缺陷并补充回归用例",
        ],
        "outcomes": [
            "回归耗时从 2 小时降到 25 分钟",
            "发现 5 个性能瓶颈",
            "核心链路自动化覆盖率提升到 75%",
            "缺陷复现效率提升 40%",
        ],
    },
    "运维开发工程师": {
        "departments": ["SRE平台组", "云原生运维组", "可观测性平台组", "基础架构部"],
        "research": ["云原生", "可观测性", "自动化运维", "容量治理"],
        "labs": ["云计算实验室", "分布式系统实验室", "网络运维实验室"],
        "projects": [
            ("Kubernetes 发布编排平台", "支持灰度、回滚和多环境发布审批。"),
            ("Prometheus 告警降噪系统", "按服务拓扑、时间窗口和标签聚合告警。"),
            ("服务器容量预测工具", "基于历史负载预测资源缺口并生成扩容建议。"),
            ("自动化巡检机器人", "定时检查证书、磁盘、水位和核心进程状态。"),
            ("日志采集链路治理平台", "监控采集延迟、丢弃率和字段解析质量。"),
            ("故障演练任务系统", "编排注入、观测和恢复动作并沉淀演练报告。"),
        ],
        "internship_actions": [
            "编写发布脚本和环境巡检任务",
            "接入指标、日志和告警规则",
            "排查容器重启、磁盘水位和网络抖动问题",
            "优化 CI/CD 流水线和镜像构建缓存",
        ],
        "outcomes": [
            "发布失败回滚时间缩短 45%",
            "无效告警减少 30%",
            "巡检问题发现提前 1 天",
            "镜像构建耗时下降 38%",
        ],
    },
    "算法工程师": {
        "departments": ["视觉算法组", "搜索排序组", "智能风控组", "数据结构算法组"],
        "research": ["图像识别", "搜索排序", "图神经网络", "深度学习", "多模态检索"],
        "labs": ["人工智能实验室", "智能计算实验室", "模式识别实验室"],
        "projects": [
            ("恶意样本图像特征识别", "将二进制片段转为灰度图并训练分类模型。"),
            ("搜索排序特征工程平台", "构建点击、文本和结构化特征并训练排序模型。"),
            ("攻击链路径推荐算法", "基于图结构推断潜在横向移动路径。"),
            ("图像验证码识别实验", "训练轻量 CNN 模型处理变形字符识别。"),
            ("相似漏洞描述检索系统", "使用语义向量和关键词召回匹配相似漏洞。"),
            ("流量异常分类模型", "提取统计特征并识别扫描、爆破和异常访问。"),
        ],
        "internship_actions": [
            "实现特征抽取、训练和离线评估流程",
            "调参对比 CNN、Transformer 和传统模型",
            "优化召回候选生成和排序特征",
            "清洗样本标签并分析误判案例",
        ],
        "outcomes": [
            "离线 AUC 提升 0.06",
            "Top5 准确率提升 10%",
            "训练数据噪声下降 20%",
            "推理耗时降低 28%",
        ],
    },
    "Java开发工程师": {
        "departments": ["Java平台组", "微服务治理组", "网关服务组", "结算系统组"],
        "research": ["JVM 调优", "微服务治理", "分布式事务", "服务网关"],
        "labs": ["软件工程实验室", "分布式系统实验室", "工程效能实验室"],
        "projects": [
            ("微服务注册治理平台", "管理服务注册、健康检查和调用限流策略。"),
            ("统一认证与单点登录系统", "支持 OAuth、角色权限和登录审计。"),
            ("分布式任务调度中心", "支持定时任务分片、重试和执行日志查询。"),
            ("网关灰度路由服务", "按租户、版本和请求头配置灰度策略。"),
            ("账单核对批处理系统", "比对多来源账单并生成差异工单。"),
            ("JVM 性能诊断工具", "采集线程、GC 和堆快照指标辅助排障。"),
        ],
        "internship_actions": [
            "实现 Spring Cloud 服务接口和熔断策略",
            "排查 JVM 内存、线程池和连接池问题",
            "编写批处理任务和补偿逻辑",
            "完善权限拦截、审计日志和单元测试",
        ],
        "outcomes": [
            "服务重启故障率下降 25%",
            "批处理窗口缩短 40%",
            "网关路由配置生效从分钟级降到秒级",
            "接口单测覆盖率提升到 70%",
        ],
    },
    "安全运营工程师": {
        "departments": ["安全运营中心", "应急响应组", "威胁情报组", "SOC平台组"],
        "research": ["威胁检测", "应急响应", "日志分析", "攻击溯源", "SIEM 规则"],
        "labs": ["网络攻防实验室", "安全运营实验室", "威胁情报实验室"],
        "projects": [
            ("SOC 告警分级处置平台", "按资产、攻击阶段和威胁情报给告警分级。"),
            ("钓鱼邮件溯源分析工具", "解析邮件头、URL 和附件特征辅助研判。"),
            ("主机入侵排查脚本集", "采集进程、计划任务、登录和网络连接证据。"),
            ("威胁情报 IOC 匹配系统", "将域名、IP、哈希与历史告警自动关联。"),
            ("安全事件复盘知识库", "沉淀处置步骤、证据链和改进项。"),
            ("日志规则命中率分析看板", "统计规则误报、漏报和处置耗时。"),
        ],
        "internship_actions": [
            "参与告警研判、证据收集和事件升级",
            "编写 SIEM 规则并分析误报样本",
            "整理 IOC 情报并关联历史日志",
            "输出应急响应复盘和处置手册",
        ],
        "outcomes": [
            "高危告警平均响应时间缩短 30%",
            "新增 20+ 条检测规则",
            "误报样本复盘形成规则优化建议",
            "完成多起模拟入侵事件闭环",
        ],
    },
}


def generate_mock_resumes(count: int, seed: int = 20260616) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    docs = []
    used_names: set[str] = set()

    for index in range(1, count + 1):
        name = _unique_name(rng, used_names)
        gender = "女" if rng.random() < 0.46 else "男"
        position_code, position_name = POSITIONS[(index - 1) % len(POSITIONS)]
        skill_pool = SKILL_POOLS[position_name]
        skills = rng.sample(skill_pool, k=min(len(skill_pool), rng.randint(4, 6)))
        if "Linux" not in skills and rng.random() < 0.4:
            skills.append("Linux")
        current_city = rng.choice(CITIES)
        expected_cities = rng.sample(CITIES, k=rng.randint(1, 2))
        if "北京" not in expected_cities and rng.random() < 0.65:
            expected_cities[0] = "北京"
        profile = POSITION_BACKGROUNDS[position_name]

        highest_degree = "硕士" if rng.random() < 0.48 else "本科"
        birth_year = rng.randint(1994, 2002)
        birth_date = date(birth_year, rng.randint(1, 12), rng.randint(1, 28))
        school, college = rng.choice(SCHOOLS)
        major = rng.choice(MAJORS)
        graduation_year = rng.randint(2022, 2026)
        graduation_date = date(graduation_year, 6, rng.choice([20, 25, 30]))
        apply_time = date(2026, rng.randint(1, 6), rng.randint(1, 28))
        candidate_no = f"M{20260000 + index:08d}"

        education = _build_education(rng, highest_degree, school, college, major, graduation_year, profile)
        internships = _build_internships(rng, index, position_name, skills, apply_time, profile)
        projects = _build_projects(rng, position_name, skills, graduation_year, profile)
        section_text = _build_section_text(name, gender, birth_date, current_city, highest_degree, school, major, education, internships, projects, skills)
        raw_text = _build_raw_text(name, candidate_no, apply_time, position_name, section_text)

        doc = {
            "resume_id": candidate_no,
            "parse_status": "ok",
            "parse_errors": [],
            "parser_version": "mock-v1",
            "file": {
                "path": f"/mock/resumes/{candidate_no}-{name}-{position_name}.pdf",
                "name": f"{candidate_no}-{name}-{position_name}.pdf",
                "sha256": _resume_id(seed, index),
                "size": rng.randint(80_000, 260_000),
                "mtime": f"{apply_time.isoformat()}T09:00:00",
                "detected_type": "mock_pdf",
                "encoding": "utf-8",
            },
            "application": {
                "candidate_no": candidate_no,
                "apply_time": apply_time.isoformat(),
                "company": "奇安信集团",
                "position_code": position_code,
                "position_name": position_name,
                "wishes": [
                    {"rank": 1, "position_name": position_name, "company": "奇安信集团"}
                ],
                "expected_work_cities": expected_cities,
            },
            "candidate": {
                "name": name,
                "gender": gender,
                "birth_date": birth_date.isoformat(),
                "ethnicity": "汉族",
                "nationality": "中国",
                "political_status": rng.choice(["团员", "群众", "中共党员"]),
                "current_city": current_city,
                "highest_degree": highest_degree,
                "graduation_date": graduation_date.isoformat(),
                "school": school,
                "major": major,
                "phone": f"13{rng.randint(100000000, 999999999)}",
                "emergency_phone": f"15{rng.randint(100000000, 999999999)}",
                "email": f"mock{index:03d}@example.com",
                "accept_transfer": rng.choice(["是", "否"]),
                "interview_city": rng.choice(expected_cities),
                "recruiting_source": rng.choice(["校园招聘官网", "员工推荐", "BOSS直聘", "牛客网"]),
            },
            "education": education,
            "internships": internships,
            "projects": projects,
            "languages": {
                "english_exam_score": rng.choice(["CET 4: 475", "CET 6: 520", "CET 6: 565", "IELTS: 6.5"]),
                "english_spoken_level": rng.choice(["可日常沟通", "可技术面试", "熟练"]),
            },
            "it_skill_items": [
                {
                    "skill_name": skill,
                    "duration": f"{rng.randint(1, 4)}年",
                    "proficiency": rng.choice(["熟练", "精通", "了解"]),
                    "primary_languages": ", ".join(skills[:2]),
                    "other_languages": ", ".join(skills[2:]),
                    "is_current": False,
                }
                for skill in skills[:3]
            ],
            "skills": skills,
            "awards": _build_awards(rng, position_name, profile),
            "offer_internship": {
                "post_graduation_intention": "就业",
                "can_intern": rng.choice(["是", "否"]),
                "available_start_date": apply_time.isoformat(),
                "weekly_workdays": rng.choice(["3天", "4天", "5天"]),
                "internship_period": rng.choice(["3个月", "6个月", "长期"]),
            },
            "uploaded_resume": {"chinese_resume": f"{name}-中文简历.pdf"},
            "section_text": section_text,
            "raw_text": raw_text,
        }
        docs.append(_enrich_doc(doc))
    add_doc_embeddings(docs)
    return docs


def import_mock_resumes(
    docs: list[dict[str, Any]],
    es_url: str,
    index: str,
    alias: str,
    recreate: bool = False,
) -> dict[str, Any]:
    target = alias if _target_exists(es_url, alias) else index

    if recreate:
        target = _versioned_index_name(index)
        _request("PUT", f"{es_url}/{target}", json_body=INDEX_BODY, ok_statuses={200})
    elif not _target_exists(es_url, target):
        _request("PUT", f"{es_url}/{index}", json_body=INDEX_BODY, ok_statuses={200})
        target = index

    if docs:
        _bulk_index(es_url, target, docs)
        _request("POST", f"{es_url}/{target}/_refresh", ok_statuses={200})

    count = _request("GET", f"{es_url}/{target}/_count", ok_statuses={200})["count"]
    if recreate or not _target_exists(es_url, alias):
        _switch_alias(es_url, target, alias)
    return {"target": target, "generated": len(docs), "indexed_total": count}


def write_jsonl(docs: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as file:
        for doc in docs:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _target_exists(es_url: str, target: str) -> bool:
    response = requests.head(f"{es_url}/{target}", timeout=10)
    return response.status_code == 200


def _unique_name(rng: random.Random, used_names: set[str]) -> str:
    while True:
        name = rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES)
        if name not in used_names:
            used_names.add(name)
            return name


def _resume_id(seed: int, index: int) -> str:
    return hashlib.sha256(f"mock-resume-{seed}-{index}".encode("utf-8")).hexdigest()


def _build_education(
    rng: random.Random,
    highest_degree: str,
    school: str,
    college: str,
    major: str,
    graduation_year: int,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    undergraduate_school, undergraduate_college = rng.choice(SCHOOLS)
    undergraduate_major = rng.choice(MAJORS)
    undergraduate = {
        "start_date": f"{graduation_year - 4}-09-01" if highest_degree == "本科" else f"{graduation_year - 7}-09-01",
        "end_date": f"{graduation_year}-06-25" if highest_degree == "本科" else f"{graduation_year - 3}-06-25",
        "school": school if highest_degree == "本科" else undergraduate_school,
        "college": college if highest_degree == "本科" else undergraduate_college,
        "major": major if highest_degree == "本科" else undergraduate_major,
        "education_level": "本科",
        "degree": "学士",
        "research_direction": rng.choice(profile["research"] + ["软件工程", "数据挖掘", "网络安全"]),
        "lab_name": rng.choice(profile["labs"]),
        "paper_level": rng.choice(["无", "校级", "EI"]),
        "start_date_raw": f"{graduation_year - 4}-09-01" if highest_degree == "本科" else f"{graduation_year - 7}-09-01",
        "end_date_raw": f"{graduation_year}-06-25" if highest_degree == "本科" else f"{graduation_year - 3}-06-25",
        "is_current": False,
    }
    if highest_degree == "本科":
        return [undergraduate]

    graduate = {
        "start_date": f"{graduation_year - 3}-09-01",
        "end_date": f"{graduation_year}-06-25",
        "school": school,
        "college": college,
        "major": major,
        "education_level": "硕士研究生",
        "degree": "硕士",
        "research_direction": rng.choice(profile["research"]),
        "lab_name": rng.choice(profile["labs"]),
        "paper_level": rng.choice(["无", "EI", "SCI", "核心期刊"]),
        "start_date_raw": f"{graduation_year - 3}-09-01",
        "end_date_raw": f"{graduation_year}-06-25",
        "is_current": False,
    }
    return [undergraduate, graduate]


def _build_internships(
    rng: random.Random,
    index: int,
    position_name: str,
    skills: list[str],
    apply_time: date,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    items = []
    internship_count = 1 if rng.random() < 0.65 else 2
    for offset in range(internship_count):
        start_year = rng.randint(2022, 2025)
        start_month = rng.randint(1, 10)
        start = date(start_year, start_month, rng.choice([1, 5, 10, 15]))
        duration_months = rng.randint(3, 10)
        end_month = min(12, start_month + duration_months)
        end_year = start_year + (start_month + duration_months - 1) // 12
        end_month = ((start_month + duration_months - 1) % 12) + 1
        end = date(end_year, end_month, rng.choice([15, 20, 25]))
        if end > apply_time or (offset == 0 and rng.random() < 0.25):
            end = None
        company = COMPANIES[(index + offset) % len(COMPANIES)]
        title = position_name.replace("工程师", "实习生") if "工程师" in position_name else f"{position_name}实习生"
        action = rng.choice(profile["internship_actions"])
        outcome = rng.choice(profile["outcomes"])
        description = (
            f"在{company}{profile['departments'][offset % len(profile['departments'])]}担任{title}，"
            f"主要负责{action}；使用{', '.join(skills[:3])}完成实现和验证，{outcome}。"
        )
        items.append(
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat() if end else None,
                "company": company,
                "company_type": rng.choice(["民营企业", "互联网企业", "国有企业", "外资企业"]),
                "company_size": rng.choice(["500-1000人", "1000-5000人", "5000人以上"]),
                "department": rng.choice(profile["departments"]),
                "title": title,
                "work_type": "实习",
                "description": description,
                "start_date_raw": start.isoformat(),
                "end_date_raw": end.isoformat() if end else "至今",
                "is_current": end is None,
            }
        )
    return items


def _build_projects(
    rng: random.Random,
    position_name: str,
    skills: list[str],
    graduation_year: int,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    templates = rng.sample(profile["projects"], k=2)
    projects = []
    for offset, (project_name, description) in enumerate(templates):
        start = date(graduation_year - 2 + offset, rng.randint(1, 8), rng.choice([1, 10, 15]))
        end = date(start.year, min(12, start.month + rng.randint(2, 5)), rng.choice([15, 20, 28]))
        action = rng.choice(profile["internship_actions"])
        outcome = rng.choice(profile["outcomes"])
        responsibility = (
            f"担任{position_name}相关角色，负责{action}；"
            f"使用{', '.join(skills[:4])}完成方案实现、联调和评估，{outcome}。"
        )
        projects.append(
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "name": project_name,
                "description": description,
                "responsibility": responsibility,
                "start_date_raw": start.isoformat(),
                "end_date_raw": end.isoformat(),
                "is_current": False,
            }
        )
    return projects


def _build_awards(rng: random.Random, position_name: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    award_names = [
        "中国大学生服务外包创新创业大赛",
        "全国大学生信息安全竞赛",
        "蓝桥杯程序设计竞赛",
        "研究生电子设计竞赛",
        "互联网+大学生创新创业大赛",
    ]
    if rng.random() < 0.35:
        return []
    return [
        {
            "has_award": "是",
            "name": rng.choice(award_names),
            "level": rng.choice(["一等奖", "二等奖", "三等奖", "优秀奖"]),
            "description": f"围绕{position_name}的{rng.choice(profile['research'])}方向完成方案设计、实现和答辩。",
            "is_current": False,
        }
    ]


def _build_section_text(
    name: str,
    gender: str,
    birth_date: date,
    current_city: str,
    highest_degree: str,
    school: str,
    major: str,
    education: list[dict[str, Any]],
    internships: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    skills: list[str],
) -> dict[str, str]:
    personal_info = "\n".join(
        [
            f"姓名: {name}",
            f"性别: {gender}",
            f"出生日期: {birth_date.isoformat()}",
            "民族: 汉族",
            "国籍: 中国",
            f"目前所在城市: {current_city}",
            f"最高学位: {highest_degree}",
            f"毕业院校: {school}",
            f"专业: {major}",
        ]
    )
    education_text = "\n".join(
        "\n".join(
            [
                f"开始日期: {item['start_date']}",
                f"取得毕业证时间: {item['end_date']}",
                f"学校: {item['school']}",
                f"学院: {item['college']}",
                f"专业: {item['major']}",
                f"学历: {item['education_level']}",
                f"学位: {item['degree']}",
                f"研究方向: {item['research_direction']}",
                f"实验室名称: {item['lab_name']}",
                f"论文发表等级: {item['paper_level']}",
            ]
        )
        for item in education
    )
    internships_text = "\n".join(
        "\n".join(
            [
                f"开始日期: {item['start_date']}",
                f"结束日期: {item['end_date_raw']}",
                f"企业名称: {item['company']}",
                f"所在部门: {item['department']}",
                f"职位名称: {item['title']}",
                f"工作性质: {item['work_type']}",
                f"工作描述: {item['description']}",
            ]
        )
        for item in internships
    )
    projects_text = "\n".join(
        "\n".join(
            [
                f"开始日期: {item['start_date']}",
                f"结束日期: {item['end_date']}",
                f"项目名称: {item['name']}",
                f"项目描述: {item['description']}",
                f"项目职责: {item['responsibility']}",
            ]
        )
        for item in projects
    )
    return {
        "personal_info": personal_info,
        "education": education_text,
        "internships": internships_text,
        "projects": projects_text,
        "languages": "英语能力: CET 4及以上，可参与技术沟通",
        "it_skills": "IT技能: " + ", ".join(skills),
    }


def _build_raw_text(
    name: str,
    candidate_no: str,
    apply_time: date,
    position_name: str,
    section_text: dict[str, str],
) -> str:
    return "\n".join(
        [
            name,
            "求职意愿:",
            f"第1志愿--{position_name}--奇安信集团",
            f"个人编号: {candidate_no}",
            f"申请时间: {apply_time.isoformat()}",
            "个人信息",
            section_text["personal_info"],
            "教育经历",
            section_text["education"],
            "实习经历",
            section_text["internships"],
            "项目经验",
            section_text["projects"],
            "语言能力",
            section_text["languages"],
            "IT技能",
            section_text["it_skills"],
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate mock resumes and import them into Elasticsearch.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--output", default="mock_resumes.jsonl")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--recreate", action="store_true", help="Recreate the index before importing mock resumes.")
    args = parser.parse_args()

    docs = generate_mock_resumes(args.count, args.seed)
    if not args.no_output:
        write_jsonl(docs, args.output)
    result = import_mock_resumes(docs, args.es_url, args.index, args.alias, recreate=args.recreate)
    if not args.no_output:
        result["output"] = str(Path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
