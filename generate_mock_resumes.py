"""确定性生成贴合真实简历字段结构的模拟简历。

字段结构对齐真实 html-doc-v1 解析输出（见 resume_parser.py 对两份奇安信真实
简历的解析结果），补全了 languages / awards / offer_internship / it_skill_items
和完整 education 子字段——这些字段参与检索与 rerank，旧的 llm-diverse-v2 mock
全为 null，导致这些信号从未被评测覆盖。

设计要点：
- 不写入 candidate.years_experience，交给 import_to_es._estimate_years_experience
  从实习时间跨度估算（与真实简历一致）。
- 岗位族贴合奇安信安全业务 + 两份真实简历的 ML/测试岗。
- 含一批 hard negative（"沾边但不对"）简历，用于考验精排和 forbidden@10。
- 固定 random seed，可复现，便于后续重标 qrels。
- 不依赖 LLM：避免"LLM 生成 + LLM 评测"的同源偏差。

用法：
    python generate_mock_resumes.py            # 写入 data/ai_generated.jsonl
    python generate_mock_resumes.py -o out.jsonl --count 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PARSER_VERSION = "mock-realistic-v2"
COMPANY = "奇安信集团"
SEED = 20260629

# 申请季：真实两份简历 apply_time 在 2019-09，这里用一个统一的招聘季，
# 让 years_experience 估算有稳定的参考截止日期。
APPLY_WINDOW_START = date(2026, 3, 1)
APPLY_WINDOW_END = date(2026, 6, 20)

CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都", "西安", "南京", "武汉", "苏州"]

FAMILY_NAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳")
GIVEN_NAMES = [
    "伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "勇", "军", "杰", "娟", "涛", "明",
    "超", "霞", "平", "刚", "桂英", "宇轩", "子涵", "梓萱", "浩然", "欣怡", "雨泽", "思源",
    "嘉豪", "晨曦", "博文", "佳", "琳", "婷", "辉", "鑫", "雷", "斌", "波", "宁", "璐",
]

SCHOOLS = [
    "清华大学", "北京大学", "浙江大学", "上海交通大学", "复旦大学", "南京大学",
    "华中科技大学", "西安电子科技大学", "电子科技大学", "北京邮电大学", "哈尔滨工业大学",
    "武汉大学", "中山大学", "四川大学", "山东大学", "北京理工大学", "东南大学",
    "西安交通大学", "同济大学", "北京航空航天大学",
]

ENGLISH_EXAM = ["CET 4: 480", "CET 4: 520", "CET 6: 436", "CET 6: 510", "CET 6: 560", "雅思 6.5", "托福 95"]
ENGLISH_SPOKEN = ["可面试", "简单会话", "熟练", "良好"]
COMPANY_TYPES = ["私营/民营企业", "国有企业", "外资企业", "合资企业", "事业单位"]

NAME_INSERT_CHARS = list("子梓雨思晨嘉昊宇泽一卓亦书若予可欣睿承彦景")

# 实习公司使用真实企业名，并按岗位族做倾斜，避免所有候选人都像在同一家
# "某科技公司" 实习。类型保持在真实解析字段可接受的枚举内。
DEFAULT_REAL_COMPANIES = [
    ("腾讯", "私营/民营企业"),
    ("阿里巴巴", "私营/民营企业"),
    ("百度", "私营/民营企业"),
    ("京东", "私营/民营企业"),
    ("美团", "私营/民营企业"),
    ("华为", "私营/民营企业"),
    ("字节跳动", "私营/民营企业"),
    ("小米集团", "私营/民营企业"),
]
REAL_COMPANIES_BY_FAMILY: dict[str, list[tuple[str, str]]] = {
    "security_research": [
        ("360数字安全集团", "私营/民营企业"),
        ("奇安信集团", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"),
        ("绿盟科技", "私营/民营企业"),
        ("启明星辰", "私营/民营企业"),
        ("天融信", "私营/民营企业"),
        ("安恒信息", "私营/民营企业"),
        ("亚信安全", "私营/民营企业"),
    ],
    "blue_team": [
        ("奇安信集团", "私营/民营企业"),
        ("360数字安全集团", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"),
        ("绿盟科技", "私营/民营企业"),
        ("启明星辰", "私营/民营企业"),
        ("安恒信息", "私营/民营企业"),
        ("亚信安全", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("阿里云", "私营/民营企业"),
    ],
    "red_team": [
        ("360数字安全集团", "私营/民营企业"),
        ("奇安信集团", "私营/民营企业"),
        ("绿盟科技", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"),
        ("安恒信息", "私营/民营企业"),
        ("启明星辰", "私营/民营企业"),
        ("天融信", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("美团", "私营/民营企业"),
    ],
    "devsecops": [
        ("阿里云", "私营/民营企业"),
        ("腾讯云", "私营/民营企业"),
        ("华为云", "私营/民营企业"),
        ("京东科技", "私营/民营企业"),
        ("蚂蚁集团", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"),
        ("美团", "私营/民营企业"),
        ("快手", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"),
    ],
    "ml_llm": [
        ("百度", "私营/民营企业"),
        ("阿里巴巴", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"),
        ("科大讯飞", "私营/民营企业"),
        ("商汤科技", "私营/民营企业"),
        ("旷视科技", "私营/民营企业"),
        ("小红书", "私营/民营企业"),
        ("美团", "私营/民营企业"),
        ("京东", "私营/民营企业"),
    ],
    "backend_go": [
        ("字节跳动", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("快手", "私营/民营企业"),
        ("京东", "私营/民营企业"),
        ("美团", "私营/民营企业"),
        ("百度", "私营/民营企业"),
        ("哔哩哔哩", "私营/民营企业"),
        ("小米集团", "私营/民营企业"),
        ("滴滴", "私营/民营企业"),
    ],
    "backend_java": [
        ("阿里巴巴", "私营/民营企业"),
        ("蚂蚁集团", "私营/民营企业"),
        ("京东", "私营/民营企业"),
        ("美团", "私营/民营企业"),
        ("招商银行", "国有企业"),
        ("平安科技", "私营/民营企业"),
        ("携程", "私营/民营企业"),
        ("贝壳找房", "私营/民营企业"),
        ("用友网络", "私营/民营企业"),
    ],
    "backend_cpp": [
        ("华为", "私营/民营企业"),
        ("中兴通讯", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("百度", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"),
        ("商汤科技", "私营/民营企业"),
        ("中科曙光", "国有企业"),
        ("海康威视", "私营/民营企业"),
        ("大华股份", "私营/民营企业"),
        ("寒武纪", "私营/民营企业"),
    ],
    "frontend": [
        ("字节跳动", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("阿里巴巴", "私营/民营企业"),
        ("百度", "私营/民营企业"),
        ("美团", "私营/民营企业"),
        ("京东", "私营/民营企业"),
        ("小红书", "私营/民营企业"),
        ("哔哩哔哩", "私营/民营企业"),
        ("携程", "私营/民营企业"),
        ("金山办公", "私营/民营企业"),
    ],
    "data_analysis": [
        ("美团", "私营/民营企业"),
        ("京东", "私营/民营企业"),
        ("阿里巴巴", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("拼多多", "私营/民营企业"),
        ("滴滴", "私营/民营企业"),
        ("小红书", "私营/民营企业"),
        ("贝壳找房", "私营/民营企业"),
        ("快手", "私营/民营企业"),
    ],
    "testing": [
        ("华为", "私营/民营企业"),
        ("腾讯", "私营/民营企业"),
        ("阿里巴巴", "私营/民营企业"),
        ("百度", "私营/民营企业"),
        ("京东", "私营/民营企业"),
        ("小米集团", "私营/民营企业"),
        ("中兴通讯", "私营/民营企业"),
        ("金山办公", "私营/民营企业"),
        ("用友网络", "私营/民营企业"),
        ("携程", "私营/民营企业"),
    ],
    "product": [
        ("腾讯", "私营/民营企业"),
        ("阿里云", "私营/民营企业"),
        ("华为云", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"),
        ("奇安信集团", "私营/民营企业"),
        ("绿盟科技", "私营/民营企业"),
        ("用友网络", "私营/民营企业"),
        ("金山办公", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"),
    ],
    "hn_search_ops": [
        ("百度", "私营/民营企业"),
        ("阿里云", "私营/民营企业"),
        ("腾讯云", "私营/民营企业"),
        ("京东科技", "私营/民营企业"),
        ("华为云", "私营/民营企业"),
    ],
    "hn_it_ops": [
        ("中国移动", "国有企业"),
        ("中国电信", "国有企业"),
        ("国家电网", "国有企业"),
        ("招商银行", "国有企业"),
        ("中信银行", "国有企业"),
        ("中国联通", "国有企业"),
    ],
    "hn_cpp_biz": [
        ("用友网络", "私营/民营企业"),
        ("金蝶软件", "私营/民营企业"),
        ("广联达", "私营/民营企业"),
        ("航天信息", "国有企业"),
        ("浪潮软件", "国有企业"),
    ],
    "hn_bi_report": [
        ("帆软软件", "私营/民营企业"),
        ("用友网络", "私营/民营企业"),
        ("金蝶软件", "私营/民营企业"),
        ("京东", "私营/民营企业"),
        ("美团", "私营/民营企业"),
    ],
}

DEFAULT_VARIATION_BANK = {
    "domains": ["企业内部平台", "多业务线协同", "区域交付项目", "研发效能平台"],
    "scales": ["接入 6 个业务系统", "覆盖 3 条核心链路", "服务 2 个试点团队", "沉淀 20+ 条流程模板"],
    "outcomes": ["将交付周期缩短 18%", "把人工核对量降低 30%", "提升跨团队协作效率", "形成可复用实施手册"],
    "actions": ["梳理关键风险点", "补齐监控与复盘流程", "推动灰度上线", "完善使用文档与验收标准"],
}
FAMILY_VARIATION_BANKS: dict[str, dict[str, list[str]]] = {
    "security_research": {
        "domains": ["浏览器内核", "云主机安全", "开源组件供应链", "终端防护引擎", "IoT 固件"],
        "scales": ["覆盖 60+ 开源组件", "累计分析 180+ 崩溃样本", "构建 12 类语料变异策略", "复现 9 个历史 CVE"],
        "outcomes": ["将漏洞复现周期从 3 天缩短到 1 天", "新增 4 条高危漏洞利用链验证", "把误报样本过滤率提升 26%", "沉淀可复用审计 checklist"],
        "actions": ["补充崩溃自动归因规则", "完善 PoC 最小化脚本", "整理补丁 diff 与利用条件", "搭建隔离复现环境"],
    },
    "blue_team": {
        "domains": ["金融终端安全", "政务云日志中心", "办公网安全运营", "勒索软件处置", "云上主机防护"],
        "scales": ["接入 12 类日志源", "日均处理 2 万条告警", "覆盖 800+ 台终端", "沉淀 70+ 条 Sigma/YARA 规则"],
        "outcomes": ["将高危告警确认时间缩短 35%", "把重复告警压降 28%", "完成 6 次攻击链复盘", "提升横向移动发现率"],
        "actions": ["补齐告警分级和流转规则", "编写攻击链时间线模板", "优化 IOC 匹配与白名单", "完善应急处置剧本"],
    },
    "red_team": {
        "domains": ["AD 域攻防演练", "互联网资产暴露面", "Web 业务纵深测试", "云上攻防对抗", "办公网钓鱼演练"],
        "scales": ["覆盖 30+ 个业务系统", "梳理 400+ 个外部资产", "构建 8 条攻击路径", "复测 50+ 个漏洞项"],
        "outcomes": ["输出 12 条高优先级加固建议", "将资产探测误报降低 22%", "形成完整攻击链报告", "沉淀自动化检查脚本"],
        "actions": ["补充权限维持风险验证", "整理横向移动路径图", "编写漏洞复测脚本", "推动修复闭环跟踪"],
    },
    "devsecops": {
        "domains": ["云原生研发流水线", "金融 DevSecOps", "研发效能平台", "容器镜像准入", "软件供应链治理"],
        "scales": ["接入 40+ 个代码仓库", "扫描 300+ 个容器镜像", "覆盖 5 条 CI/CD 流水线", "治理 1200+ 个依赖组件"],
        "outcomes": ["将高危组件暴露时长降低 42%", "把误报工单压降 25%", "将安全卡点前移到合并请求阶段", "提升漏洞修复闭环率"],
        "actions": ["设计风险分级策略", "补充 SCA 白名单治理", "接入制品库准入控制", "输出研发侧修复指引"],
    },
    "ml_llm": {
        "domains": ["企业知识库", "智能客服", "安全运营问答", "内容理解平台", "搜索推荐场景"],
        "scales": ["处理 80 万条问答语料", "接入 20+ 类业务文档", "构建 5000 万级向量索引", "支持 6 个知识主题"],
        "outcomes": ["将 Top3 召回率提升 11%", "把人工标注成本降低 24%", "将问答命中率提升到 82%", "缩短离线评测耗时 40%"],
        "actions": ["设计负样本挖掘策略", "优化 chunk 切分和重排特征", "补齐 hallucination 评测集", "搭建离线评测看板"],
    },
    "backend_go": {
        "domains": ["云原生网关", "实时消息平台", "容器调度平台", "交易风控服务", "可观测性平台"],
        "scales": ["支撑 10 万 QPS 峰值", "管理 3000+ 个 Pod", "接入 80+ 个微服务", "处理 PB 级日志索引"],
        "outcomes": ["将 P99 延迟降低 31%", "把发布回滚耗时缩短 45%", "提升服务可用性到 99.95%", "降低资源成本 18%"],
        "actions": ["优化连接池和批量写入", "补齐限流熔断策略", "重构控制器 reconcile 流程", "接入 Prometheus 告警"],
    },
    "backend_java": {
        "domains": ["金融交易链路", "电商订单中心", "会员权益平台", "支付清结算系统", "企业中台"],
        "scales": ["支撑 8 万 QPS 峰值", "处理千万级订单数据", "接入 30+ 个下游系统", "拆分 18 个核心服务"],
        "outcomes": ["将接口 P95 延迟降低 27%", "把慢 SQL 数量减少 38%", "提升核心链路可用性", "降低重复开发成本"],
        "actions": ["治理分布式事务边界", "优化缓存穿透保护", "补齐链路追踪埋点", "设计灰度发布方案"],
    },
    "backend_cpp": {
        "domains": ["行情接入网关", "实时风控引擎", "存储引擎内核", "音视频传输链路", "边缘计算节点"],
        "scales": ["单节点处理 120 万条消息/秒", "压测 40Gbps 网络流量", "管理 20TB 热数据", "覆盖 6 类低延迟场景"],
        "outcomes": ["将 P99 延迟降低 33%", "把 CPU 占用降低 21%", "提升吞吐稳定性", "缩短故障定位时间"],
        "actions": ["调整 NUMA 与绑核策略", "优化内存池复用", "补充 perf 火焰图分析", "重构异步 IO 路径"],
    },
    "frontend": {
        "domains": ["安全态势大屏", "地图可视化引擎", "数据分析工作台", "低代码运营后台", "数字孪生场景"],
        "scales": ["渲染 10 万级图元", "复用 40+ 个业务组件", "覆盖 6 个运营页面", "支撑 3D 场景实时交互"],
        "outcomes": ["将首屏时间降低 32%", "把大屏帧率稳定到 55fps+", "减少重复组件开发", "提升复杂筛选交互效率"],
        "actions": ["拆分渲染层与业务状态", "优化虚拟滚动和懒加载", "封装地图图层管理", "补齐组件文档和示例"],
    },
    "data_analysis": {
        "domains": ["电商增长", "本地生活转化", "内容推荐运营", "会员留存", "营销投放归因"],
        "scales": ["分析 3000 万级用户行为", "维护 120+ 个核心指标", "支持 50+ 个 A/B 实验", "接入 8 类埋点数据"],
        "outcomes": ["将实验结论产出周期缩短 40%", "识别 3 个高价值增长策略", "提升关键漏斗转化", "减少指标口径争议"],
        "actions": ["设计分层指标体系", "校验实验分流均衡性", "搭建异动归因看板", "输出业务复盘报告"],
    },
    "testing": {
        "domains": ["移动端回归测试", "接口自动化平台", "交易链路压测", "嵌入式设备测试", "持续集成质量门禁"],
        "scales": ["维护 600+ 条自动化用例", "覆盖 20+ 个核心接口", "支撑 5 万并发压测", "接入 12 条发布流水线"],
        "outcomes": ["将回归耗时从 2 天降到 4 小时", "把线上缺陷逃逸率降低 19%", "提升用例稳定性", "提前发现容量瓶颈"],
        "actions": ["优化失败重试和截图留证", "补齐接口契约校验", "设计混合压测场景", "接入质量趋势看板"],
    },
    "product": {
        "domains": ["零信任访问", "SASE 产品", "安全运营平台", "数据产品看板", "企业协同工具"],
        "scales": ["访谈 25 位目标用户", "梳理 40+ 个需求项", "推动 3 个版本迭代", "覆盖 5 类核心使用场景"],
        "outcomes": ["将需求评审返工降低 30%", "提升试点客户激活率", "明确产品北极星指标", "缩短跨团队对齐周期"],
        "actions": ["拆解版本里程碑", "输出竞品分析和原型", "跟进灰度反馈闭环", "统一埋点和验收口径"],
    },
    "hn_search_ops": {
        "domains": ["日志检索集群", "站内搜索运维", "ES 容量治理", "索引生命周期管理", "检索监控平台"],
        "scales": ["管理 20+ 个索引集群", "日写入 5TB 日志", "维护 300+ 个索引模板", "支持 10 个业务租户"],
        "outcomes": ["将查询超时率降低 23%", "把扩容排障时间缩短 30%", "提升索引稳定性", "减少热点分片问题"],
        "actions": ["优化分片与副本策略", "补齐慢查询监控", "治理索引生命周期", "整理容量预估模板"],
    },
    "hn_it_ops": {
        "domains": ["办公网络运维", "机房基础设施", "服务器巡检", "内部系统监控", "桌面终端支持"],
        "scales": ["维护 500+ 台办公终端", "巡检 80+ 台服务器", "接入 200+ 个监控项", "处理 1000+ 条工单"],
        "outcomes": ["将普通故障响应时间缩短 20%", "提升巡检覆盖率", "减少重复工单", "完善资产台账"],
        "actions": ["规范告警升级流程", "整理网络变更记录", "补齐备份巡检清单", "优化工单分类"],
    },
    "hn_cpp_biz": {
        "domains": ["桌面业务客户端", "企业管理软件", "本地数据采集工具", "报表客户端", "窗口控件库"],
        "scales": ["维护 30+ 个业务表单", "支持 6 个桌面模块", "修复 120+ 个客户端缺陷", "适配 4 类外设"],
        "outcomes": ["提升客户端稳定性", "减少表单录入错误", "缩短版本发布周期", "改善安装包兼容性"],
        "actions": ["重构表单校验逻辑", "补齐异常日志上报", "优化安装升级流程", "完善控件复用规范"],
    },
    "hn_bi_report": {
        "domains": ["经营报表中心", "财务固定报表", "销售数据集市", "数据仓库 ETL", "BI 权限治理"],
        "scales": ["维护 90+ 张固定报表", "接入 15 个源系统", "日调度 200+ 个 ETL 任务", "服务 8 个业务部门"],
        "outcomes": ["将报表出数延迟降低 25%", "减少口径核对返工", "提升调度成功率", "完善权限审批流程"],
        "actions": ["治理维表口径", "优化 ETL 依赖关系", "补齐数据质量校验", "整理报表使用说明"],
    },
}

# 岗位族 profile。每个族定义岗位名候选、专业、研究方向、实验室、技能池、
# 项目模板、实习模板、典型证书/奖项主题。description / responsibility 文本
# 刻意不含公司名/学校名/城市（import 的语义清洗会剔除实体，生成侧也保持纯净）。
POSITION_PROFILES: dict[str, dict[str, Any]] = {
    "security_research": {
        "weight": 20,
        "positions": ["安全研究员", "安全研究员 (漏洞挖掘)", "高级安全研究员", "安全研究员 (逆向分析)"],
        "position_codes": ["S0101", "S0102", "S0103"],
        "majors": ["网络空间安全", "信息安全", "计算机科学与技术", "软件工程"],
        "research_directions": ["二进制漏洞挖掘", "模糊测试与符号执行", "内核安全", "软件供应链安全"],
        "labs": ["系统安全实验室", "网络攻防实验室", "可信计算重点实验室"],
        "skills_pool": ["C", "C++", "Python", "IDA Pro", "Ghidra", "GDB", "汇编", "Linux", "WinDbg"],
        "core_skills": ["C", "C++", "Python"],
        "projects": [
            ("浏览器引擎漏洞挖掘", "针对主流浏览器 JS 引擎进行模糊测试，构建覆盖率引导的 fuzzer 并复现内存破坏漏洞。",
             "1. 基于 AFL++ 定制 JS 引擎 fuzzer；2. 编写语料变异策略提升覆盖率；3. 分析崩溃样本定位 UAF 与类型混淆漏洞；4. 编写 PoC 并完成漏洞利用链。"),
            ("内核提权漏洞研究", "对开源内核组件做静态审计与动态测试，挖掘本地提权漏洞并分析利用面。",
             "1. 内核模块代码审计；2. 构建符号执行约束求解定位越界写；3. 编写 exploit 绕过 SMEP/SMAP；4. 输出漏洞分析报告。"),
            ("软件供应链组件审计", "对第三方开源依赖做漏洞挖掘与污点分析，建立组件风险画像。",
             "1. 依赖图谱构建；2. 污点传播分析定位注入点；3. 复现并验证 CVE；4. 形成修复建议。"),
        ],
        "internships": [
            ("安全研究部", "安全研究实习生", "参与漏洞挖掘与 PoC 编写，跟进 fuzzer 平台建设与崩溃样本分析。"),
            ("漏洞研究团队", "二进制安全实习生", "负责逆向分析与漏洞复现，协助构建自动化测试环境。"),
        ],
        "awards": [("全国大学生信息安全竞赛", "一等奖", "面向真实系统的漏洞挖掘与攻防竞赛"),
                   ("强网杯网络安全挑战赛", "二等奖", "CTF 综合攻防赛")],
    },
    "blue_team": {
        "weight": 18,
        "positions": ["蓝队应急响应工程师", "安全运营工程师 (SOC)", "威胁狩猎工程师", "安全分析师"],
        "position_codes": ["S0201", "S0202", "S0203"],
        "majors": ["网络空间安全", "信息安全", "计算机科学与技术", "网络工程"],
        "research_directions": ["威胁检测与响应", "日志分析与取证", "ATT&CK 威胁建模"],
        "labs": ["安全运营实验室", "威胁情报实验室"],
        "skills_pool": ["Python", "Splunk", "ELK", "Suricata", "Wireshark", "SIEM", "Linux", "Sigma", "YARA"],
        "core_skills": ["Python", "SIEM", "ELK"],
        "projects": [
            ("企业安全事件应急响应平台", "构建覆盖日志采集、关联分析与自动化响应的应急平台，支撑勒索软件溯源。",
             "1. 多源日志采集与归一化；2. 基于 ATT&CK 编写检测规则；3. 威胁狩猎与横向移动溯源；4. 自动化处置剧本编排。"),
            ("威胁狩猎与日志取证系统", "面向终端与流量日志做异常行为狩猎，沉淀取证分析流程。",
             "1. 终端 EDR 日志接入；2. Sigma 规则编写与误报治理；3. 失陷主机取证；4. 攻击链还原与报告输出。"),
        ],
        "internships": [
            ("安全运营中心", "安全运营实习生", "参与告警分析与规则编写，跟进应急响应值守与威胁狩猎。"),
            ("应急响应团队", "应急响应实习生", "负责日志取证与攻击链还原，协助编写处置剧本。"),
        ],
        "awards": [("网络安全应急响应技能大赛", "三等奖", "面向真实攻击场景的应急处置比赛")],
    },
    "red_team": {
        "weight": 18,
        "positions": ["红队攻防工程师", "渗透测试工程师", "高级渗透测试工程师", "红队开发工程师"],
        "position_codes": ["S0301", "S0302", "S0303"],
        "majors": ["网络空间安全", "信息安全", "计算机科学与技术"],
        "research_directions": ["内网渗透与横向移动", "Web 安全", "免杀与对抗"],
        "labs": ["攻防对抗实验室", "红队技术实验室"],
        "skills_pool": ["Python", "Go", "Cobalt Strike", "Burp Suite", "Metasploit", "C#", "PowerShell", "Linux", "内网渗透"],
        "core_skills": ["Python", "内网渗透"],
        "projects": [
            ("红队内网渗透演练平台", "建设面向 AD 域的内网横向移动与权限维持演练环境，沉淀攻击手法库。",
             "1. AD 域环境搭建；2. 横向移动与凭据窃取；3. C2 通信隐蔽通道设计；4. 权限维持与痕迹清理。"),
            ("Web 应用渗透测试", "对业务系统做黑盒与灰盒渗透，输出漏洞利用链与加固建议。",
             "1. 资产测绘与信息收集；2. 注入、越权、反序列化漏洞挖掘；3. 权限提升；4. 渗透报告与复测。"),
        ],
        "internships": [
            ("红队", "红队实习生", "参与渗透测试与攻防演练，协助开发自动化渗透工具。"),
            ("渗透测试团队", "渗透测试实习生", "负责 Web 与内网渗透，跟进漏洞复现与报告撰写。"),
        ],
        "awards": [("强网杯网络安全挑战赛", "三等奖", "CTF 攻防综合赛")],
    },
    "devsecops": {
        "weight": 15,
        "positions": ["DevSecOps 工程师", "安全平台开发工程师", "高级安全架构师 (DevSecOps)", "应用安全工程师"],
        "position_codes": ["S0401", "S0402", "S0403"],
        "majors": ["计算机科学与技术", "软件工程", "网络空间安全"],
        "research_directions": ["软件供应链安全", "CI/CD 安全", "云原生安全"],
        "labs": ["应用安全实验室", "DevSecOps 实验室"],
        "skills_pool": ["Java", "Go", "Python", "SAST", "SCA", "IAST", "Docker", "Kubernetes", "Jenkins", "GitLab CI"],
        "core_skills": ["SAST", "SCA", "Docker"],
        "projects": [
            ("DevSecOps 安全左移平台", "建设贯穿研发流程的代码安全检测平台，集成 SAST/SCA/IAST 能力。",
             "1. CI/CD 流水线安全卡点设计；2. SAST 引擎集成与误报治理；3. 第三方组件 SCA 与许可证合规；4. IAST 运行时插桩。"),
            ("容器与镜像安全扫描系统", "面向容器镜像做漏洞扫描与基线核查，支撑云原生安全。",
             "1. 镜像分层漏洞扫描；2. 基线合规检查；3. 准入控制策略；4. 风险报告与修复闭环。"),
        ],
        "internships": [
            ("安全平台研发部", "安全研发实习生", "参与安全检测引擎开发与流水线集成，跟进规则治理。"),
        ],
        "awards": [("全国大学生软件测试大赛", "二等奖", "软件质量与安全测试竞赛")],
    },
    "ml_llm": {
        "weight": 22,
        "positions": ["机器学习工程师", "算法工程师", "机器学习工程师 (LLM方向)", "NLP 算法工程师", "大模型应用工程师"],
        "position_codes": ["A0009", "A0010", "A0011"],
        "majors": ["计算机技术", "计算机科学与技术", "人工智能", "模式识别与智能系统", "软件工程"],
        "research_directions": ["自然语言处理与推荐系统", "大语言模型与检索增强", "知识图谱与问答", "深度学习"],
        "labs": ["智能信息处理实验室", "自然语言处理实验室", "机器学习重点实验室"],
        "skills_pool": ["Python", "PyTorch", "TensorFlow", "NLP", "RAG", "LangChain", "向量检索", "大模型", "MySQL", "Spark"],
        "core_skills": ["Python", "PyTorch", "NLP"],
        "projects": [
            ("企业知识库 RAG 问答系统", "基于检索增强生成构建企业知识库问答，覆盖文档解析、向量检索与召回排序。",
             "1. 文档切片与向量化入库；2. 召回与重排序链路设计；3. Prompt 工程与答案生成；4. RAG 评测与长文本处理优化。"),
            ("机器人智能导诊项目", "利用知识库与自然语言处理技术开发智能导诊，解决科室推荐与意图识别。",
             "1. 知识库构建；2. CNN 模型解决科室推荐；3. BiLSTM+CRF 做实体抽取；4. 意图识别与对话管理。"),
            ("推荐系统召回排序优化", "面向信息流场景做多路召回与精排建模，提升点击与转化。",
             "1. 多路召回链路设计；2. 特征工程与样本构造；3. 排序模型训练与上线；4. AB 实验与指标分析。"),
        ],
        "internships": [
            ("算法部", "算法工程师", "参与模型训练与上线，跟进检索召回与排序链路优化。"),
            ("机器学习平台部", "机器学习实习生", "负责数据处理与模型实验，协助大模型微调与评测。"),
        ],
        "awards": [("中国高校计算机大赛人工智能创意赛", "一等奖", "人工智能应用创新竞赛"),
                   ("研究生数学建模竞赛", "二等奖", "数学建模与算法实践")],
    },
    "backend_go": {
        "weight": 18,
        "positions": ["后端开发工程师 (Go)", "Go 云原生工程师", "高级后端工程师 (Go)", "云原生基础设施工程师"],
        "position_codes": ["B0501", "B0502", "B0503"],
        "majors": ["计算机科学与技术", "软件工程", "通信工程"],
        "research_directions": ["分布式系统", "云原生架构", "高并发服务"],
        "labs": ["分布式系统实验室", "云计算实验室"],
        "skills_pool": ["Go", "Kubernetes", "gRPC", "Docker", "etcd", "Redis", "MySQL", "Prometheus", "微服务"],
        "core_skills": ["Go", "Kubernetes", "gRPC"],
        "projects": [
            ("Kubernetes Operator 与服务网格", "基于 Operator 模式建设云原生基础设施，扩展自定义控制器与服务治理。",
             "1. 自定义资源与控制器开发；2. 服务网格流量治理；3. gRPC 微服务通信；4. 可观测性与告警接入。"),
            ("高并发网关服务", "构建高吞吐 API 网关，支撑限流熔断与灰度发布。",
             "1. 网关路由与插件化设计；2. 限流熔断与降级；3. 连接池与性能调优；4. 灰度与全链路压测。"),
        ],
        "internships": [
            ("基础架构部", "后端开发实习生", "参与微服务开发与容器化部署，跟进网关与中间件优化。"),
        ],
        "awards": [("ACM-ICPC 区域赛", "铜奖", "程序设计竞赛")],
    },
    "backend_java": {
        "weight": 18,
        "positions": ["后端开发工程师 (Java)", "Java 架构师", "资深 Java 架构师", "高级 Java 开发工程师"],
        "position_codes": ["B0601", "B0602", "B0603"],
        "majors": ["计算机科学与技术", "软件工程", "信息管理与信息系统"],
        "research_directions": ["分布式事务", "高可用架构", "中间件"],
        "labs": ["软件工程实验室", "分布式系统实验室"],
        "skills_pool": ["Java", "Spring Boot", "Spring Cloud", "MySQL", "Redis", "RocketMQ", "Kafka", "JVM", "分库分表"],
        "core_skills": ["Java", "Spring Boot", "JVM"],
        "projects": [
            ("金融级高并发交易系统", "建设金融核心链路，支撑高并发、高可用与分布式事务一致性。",
             "1. 分库分表与读写分离；2. 分布式事务与最终一致性；3. JVM 调优与 GC 优化；4. 全链路压测与容灾。"),
            ("企业级微服务中台", "基于 Spring Cloud 构建业务中台，沉淀通用服务能力。",
             "1. 服务拆分与治理；2. 配置中心与注册中心；3. 缓存与消息解耦；4. 监控与链路追踪。"),
        ],
        "internships": [
            ("交易研发部", "Java 开发实习生", "参与交易链路开发与性能优化，跟进分布式事务设计。"),
        ],
        "awards": [("蓝桥杯软件大赛", "二等奖", "程序设计竞赛")],
    },
    "backend_cpp": {
        "weight": 14,
        "positions": ["后端开发工程师 (C++)", "C++ 低延迟工程师", "高级 C++ 研发架构师", "C++ 系统工程师"],
        "position_codes": ["B0701", "B0702", "B0703"],
        "majors": ["计算机科学与技术", "电子与通信工程", "软件工程"],
        "research_directions": ["低延迟系统", "高性能计算", "网络编程"],
        "labs": ["高性能计算实验室", "网络与系统实验室"],
        "skills_pool": ["C++", "C", "RDMA", "无锁编程", "Linux", "DPDK", "Redis", "TCP/IP", "性能调优"],
        "core_skills": ["C++", "RDMA", "无锁编程"],
        "projects": [
            ("超低延迟交易基础设施", "建设高频交易实时系统，优化网络收发与内核旁路路径。",
             "1. 无锁队列与内存池设计；2. RDMA 与内核旁路收发；3. CPU 亲和与 NUMA 优化；4. 微秒级延迟剖析。"),
            ("高性能存储引擎", "实现高吞吐 KV 存储引擎，优化并发与持久化。",
             "1. LSM 存储结构实现；2. 并发控制与锁优化；3. WAL 与崩溃恢复；4. 压测与性能调优。"),
        ],
        "internships": [
            ("核心系统部", "C++ 开发实习生", "参与低延迟模块开发，跟进性能剖析与无锁优化。"),
        ],
        "awards": [("ACM-ICPC 区域赛", "银奖", "程序设计竞赛")],
    },
    "frontend": {
        "weight": 16,
        "positions": ["前端开发工程师", "高级前端开发工程师", "前端可视化工程师", "Web 前端工程师"],
        "position_codes": ["F0801", "F0802", "F0803"],
        "majors": ["计算机科学与技术", "软件工程", "数字媒体技术"],
        "research_directions": ["前端工程化", "数据可视化", "WebGL 渲染"],
        "labs": ["人机交互实验室", "可视化实验室"],
        "skills_pool": ["JavaScript", "TypeScript", "Vue", "React", "WebGL", "Three.js", "Vite", "CSS", "HTML"],
        "core_skills": ["JavaScript", "TypeScript", "Vue"],
        "projects": [
            ("大屏数据可视化平台", "基于 WebGL 与 Three.js 建设三维可视化大屏，支撑地图引擎与交互。",
             "1. WebGL 渲染管线与 shader 编写；2. 地图引擎与图层管理；3. 大数据量渲染性能优化；4. 交互与动效设计。"),
            ("企业级前端中台", "构建组件库与微前端架构，提升研发效率。",
             "1. 组件库设计与文档；2. 微前端拆分与通信；3. 构建工具链优化；4. 性能监控与首屏优化。"),
        ],
        "internships": [
            ("前端研发部", "前端开发实习生", "参与可视化组件开发与性能优化，跟进渲染管线建设。"),
        ],
        "awards": [("中国大学生计算机设计大赛", "二等奖", "软件应用与开发竞赛")],
    },
    "data_analysis": {
        "weight": 15,
        "positions": ["数据分析师", "数据分析师 (增长)", "高级数据分析师", "数据产品经理"],
        "position_codes": ["D0901", "D0902", "D0903"],
        "majors": ["统计学", "应用统计", "数据科学与大数据技术", "计算机科学与技术"],
        "research_directions": ["因果推断与实验设计", "用户增长", "指标体系建设"],
        "labs": ["数据科学实验室", "统计与计算实验室"],
        "skills_pool": ["SQL", "Python", "A/B测试", "数据可视化", "Tableau", "Hive", "Spark", "因果推断", "指标体系"],
        "core_skills": ["SQL", "Python", "A/B测试"],
        "projects": [
            ("用户增长 A/B 实验平台", "建设实验平台支撑增长策略，覆盖指标体系、归因与留存分析。",
             "1. 实验分流与样本量估算；2. 指标体系与归因模型；3. 留存与漏斗分析；4. 因果推断纠偏。"),
            ("经营分析指标看板", "搭建经营分析数据看板，沉淀核心指标与异动归因。",
             "1. 数据建模与口径治理；2. 指标计算与可视化；3. 异动检测与归因；4. 自助分析能力建设。"),
        ],
        "internships": [
            ("数据分析部", "数据分析实习生", "参与指标建设与实验分析，跟进增长策略评估。"),
        ],
        "awards": [("全国大学生统计建模大赛", "二等奖", "统计分析与建模竞赛")],
    },
    "testing": {
        "weight": 12,
        "positions": ["测试工程师", "自动化测试工程师", "高级测试开发工程师", "性能测试工程师"],
        "position_codes": ["A0014", "A0015", "A0016"],
        "majors": ["电子与通信工程", "计算机科学与技术", "软件工程", "通信工程"],
        "research_directions": ["自动化测试", "性能测试", "测试平台建设"],
        "labs": ["软件质量实验室", "嵌入式系统实验室"],
        "skills_pool": ["Python", "C", "C++", "MATLAB", "Selenium", "JMeter", "pytest", "MySQL", "Linux"],
        "core_skills": ["Python", "自动化测试"],
        "projects": [
            ("自动化测试平台", "建设覆盖接口与 UI 的自动化测试平台，提升回归效率。",
             "1. 用例管理与调度；2. 接口与 UI 自动化框架；3. 持续集成接入；4. 测试报告与覆盖率统计。"),
            ("性能压测与调优项目", "对核心服务做性能压测与瓶颈定位，输出调优方案。",
             "1. 压测场景设计；2. 性能监控与瓶颈定位；3. 调优方案验证；4. 容量评估。"),
        ],
        "internships": [
            ("测试部", "测试开发实习生", "参与自动化测试框架开发与性能压测，跟进质量度量。"),
        ],
        "awards": [("全国大学生软件测试大赛", "三等奖", "软件测试技能竞赛")],
    },
    "product": {
        "weight": 10,
        "positions": ["安全产品经理", "技术项目经理 (TPM)", "高级产品经理", "产品经理 (数据方向)"],
        "position_codes": ["P1001", "P1002", "P1003"],
        "majors": ["信息管理与信息系统", "软件工程", "计算机科学与技术", "工商管理"],
        "research_directions": ["安全产品设计", "项目管理", "数据产品"],
        "labs": ["信息系统实验室"],
        "skills_pool": ["需求分析", "Axure", "SQL", "项目管理", "SASE", "ZTNA", "零信任", "数据产品", "Visio"],
        "core_skills": ["需求分析", "项目管理"],
        "projects": [
            ("零信任安全产品规划", "负责 SASE/ZTNA 零信任访问产品的规划与落地，对接 SOC 态势感知。",
             "1. 市场调研与竞品分析；2. 产品需求与原型设计；3. 零信任访问策略设计；4. 上线与运营复盘。"),
            ("安全运营平台项目管理", "统筹安全运营平台的研发交付，协调多方资源保障里程碑。",
             "1. 需求拆解与排期；2. 跨团队协调；3. 风险识别与管控；4. 交付验收与复盘。"),
        ],
        "internships": [
            ("产品部", "产品实习生", "参与需求调研与原型设计，跟进项目排期与验收。"),
        ],
        "awards": [("全国大学生创新创业大赛", "三等奖", "创新创业项目竞赛")],
    },
}

# Hard negative 族：与某些热门检索主题"沾边但不对"。技能/文本会命中泛 token
# （检索、日志、C++、SQL 等），但缺乏目标主题的核心能力。用于考验精排与
# forbidden@10——这些人不该排在对应语义查询的前排。
HARD_NEGATIVE_PROFILES: dict[str, dict[str, Any]] = {
    "hn_search_ops": {  # 像 RAG，实为传统搜索/ES 运维
        "weight": 6,
        "positions": ["搜索运维工程师", "ES 运维工程师", "搜索平台运维"],
        "position_codes": ["H0101"],
        "majors": ["计算机科学与技术", "软件工程"],
        "research_directions": ["搜索引擎运维", "日志检索平台"],
        "labs": ["信息检索实验室"],
        "skills_pool": ["Elasticsearch", "Lucene", "Java", "Linux", "向量数据库运维", "Logstash", "Kibana", "检索"],
        "core_skills": ["Elasticsearch", "检索"],
        "projects": [
            ("企业搜索与日志检索平台运维", "负责搜索集群与向量数据库的部署运维，保障检索服务稳定。",
             "1. 检索集群部署与扩容；2. 索引分片与查询性能调优；3. 向量数据库运维监控；4. 故障排查与容量规划。（不涉及大模型与 RAG 生成）"),
        ],
        "internships": [("运维部", "搜索运维实习生", "参与检索集群运维与索引优化，跟进监控告警。")],
        "awards": [],
    },
    "hn_it_ops": {  # 像蓝队，实为普通运维/网管
        "weight": 6,
        "positions": ["IT 运维工程师", "网络运维工程师", "系统运维工程师"],
        "position_codes": ["H0201"],
        "majors": ["网络工程", "计算机科学与技术"],
        "research_directions": ["IT 基础设施运维", "网络监控"],
        "labs": ["网络工程实验室"],
        "skills_pool": ["Linux", "Shell", "Zabbix", "日志", "监控", "网络运维", "MySQL", "Nginx"],
        "core_skills": ["Linux", "监控"],
        "projects": [
            ("企业 IT 基础设施监控运维", "负责服务器与网络设备的日常运维与监控告警，保障业务可用性。",
             "1. 服务器与网络设备运维；2. 日志采集与监控大盘；3. 告警处置与值守；4. 备份与巡检。（不涉及威胁狩猎与攻击取证）"),
        ],
        "internships": [("运维部", "运维实习生", "参与日常运维与监控，跟进故障处理。")],
        "awards": [],
    },
    "hn_cpp_biz": {  # 像低延迟 C++，实为普通 C++ 业务开发
        "weight": 6,
        "positions": ["C++ 业务开发工程师", "桌面应用开发工程师 (C++)", "C++ 软件工程师"],
        "position_codes": ["H0301"],
        "majors": ["计算机科学与技术", "软件工程"],
        "research_directions": ["桌面应用开发", "业务系统开发"],
        "labs": ["软件工程实验室"],
        "skills_pool": ["C++", "Qt", "MySQL", "Windows", "MFC", "业务开发"],
        "core_skills": ["C++", "Qt"],
        "projects": [
            ("桌面客户端业务系统", "基于 Qt 开发桌面客户端，实现业务表单与数据管理功能。",
             "1. 界面与交互开发；2. 业务逻辑与表单校验；3. 本地数据库读写；4. 打包发布。（不涉及低延迟、RDMA、无锁等高性能场景）"),
        ],
        "internships": [("研发部", "C++ 开发实习生", "参与桌面客户端功能开发，跟进缺陷修复。")],
        "awards": [],
    },
    "hn_bi_report": {  # 像数据分析，实为 BI 报表开发
        "weight": 6,
        "positions": ["BI 报表开发工程师", "报表开发工程师", "数仓报表开发"],
        "position_codes": ["H0401"],
        "majors": ["信息管理与信息系统", "计算机科学与技术"],
        "research_directions": ["报表开发", "数据仓库"],
        "labs": ["信息系统实验室"],
        "skills_pool": ["SQL", "Tableau", "PowerBI", "ETL", "Excel", "数据仓库", "Kettle"],
        "core_skills": ["SQL", "Tableau"],
        "projects": [
            ("经营报表与数据仓库开发", "负责数据仓库 ETL 与固定报表开发，输出经营报表。",
             "1. ETL 抽取与清洗；2. 数据仓库分层建模；3. 固定报表与看板开发；4. 报表性能优化。（不涉及 A/B 实验、因果推断与增长分析）"),
        ],
        "internships": [("数据部", "报表开发实习生", "参与报表开发与 ETL 维护，跟进数据核对。")],
        "awards": [],
    },
}

DEGREE_CHOICES = ["本科", "硕士", "硕士", "博士"]  # 加权：硕士偏多
DEGREE_LEVEL_LABEL = {"本科": ("本科", "学士"), "硕士": ("硕士研究生", "硕士"), "博士": ("博士研究生", "博士")}
PAPER_LEVELS = ["无", "无", "EI", "SCI", "核心期刊", "CCF-B"]
INTERN_PERIODS = ["3个月以上", "6个月以上", "实习期不限"]


def _rng_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 0)))


def _fmt(d: date) -> str:
    return d.isoformat()


def _phone(rng: random.Random) -> str:
    return "1" + rng.choice("3578") + "".join(str(rng.randint(0, 9)) for _ in range(9))


def _email(rng: random.Random, pinyin_seed: str) -> str:
    domains = ["126.com", "163.com", "qq.com", "gmail.com", "foxmail.com", "outlook.com"]
    handle = "user" + hashlib.md5(pinyin_seed.encode()).hexdigest()[:8]
    return f"{handle}@{rng.choice(domains)}"


def _name(rng: random.Random) -> str:
    return rng.choice(FAMILY_NAMES) + rng.choice(GIVEN_NAMES)


def _unique_name(rng: random.Random, used_names: set[str], idx: int) -> str:
    """保持姓名像真实中文名，同时避免 mock 数据里短列表抽样导致撞名。"""
    base = _name(rng)
    if base not in used_names:
        used_names.add(base)
        return base
    family = base[0]
    given = base[1:]
    for offset in range(len(NAME_INSERT_CHARS)):
        insert = NAME_INSERT_CHARS[(idx + offset) % len(NAME_INSERT_CHARS)]
        candidate = family + insert + given
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    candidate = f"{family}{given}{idx}"
    used_names.add(candidate)
    return candidate


def _variation(rng: random.Random, family: str) -> dict[str, str]:
    bank = FAMILY_VARIATION_BANKS.get(family, DEFAULT_VARIATION_BANK)
    return {
        "domain": rng.choice(bank["domains"]),
        "scale": rng.choice(bank["scales"]),
        "outcome": rng.choice(bank["outcomes"]),
        "action": rng.choice(bank["actions"]),
    }


def _real_company(rng: random.Random, family: str, used_companies: set[str]) -> tuple[str, str]:
    catalog = REAL_COMPANIES_BY_FAMILY.get(family, DEFAULT_REAL_COMPANIES)
    available = [item for item in catalog if item[0] not in used_companies] or catalog
    company, company_type = rng.choice(available)
    used_companies.add(company)
    return company, company_type


def _personalized_project(
    rng: random.Random,
    family: str,
    name: str,
    desc: str,
    resp: str,
    skills: list[str],
) -> tuple[str, str, str]:
    var = _variation(rng, family)
    skill_hint = "、".join(skills[:2]) if skills else rng.choice(["Python", "SQL", "Linux"])
    name_style = rng.choice(["plain", "domain_suffix", "domain_prefix"])
    if name_style == "domain_suffix":
        project_name = f"{name}（{var['domain']}）"
    elif name_style == "domain_prefix":
        project_name = f"{var['domain']}{name}"
    else:
        project_name = name
    project_desc = f"{desc} 面向{var['domain']}场景，{var['scale']}。"
    project_resp = f"{resp.rstrip('。')}；5. 基于 {skill_hint} {var['action']}，{var['outcome']}。"
    return project_name, project_desc, project_resp


def _personalized_internship_desc(
    rng: random.Random,
    family: str,
    desc: str,
    company: str,
    skills: list[str],
) -> str:
    var = _variation(rng, family)
    skill_hint = "、".join(skills[:2]) if skills else rng.choice(["Python", "SQL", "Linux"])
    return (
        f"{desc} 在{company}的{var['domain']}场景中，"
        f"使用 {skill_hint} {var['action']}，{var['outcome']}。"
    )


def _build_education(rng: random.Random, profile: dict, degree: str, grad_year: int) -> list[dict]:
    """构造教育经历：本科一段；硕士/博士再加一段（含研究方向/实验室/论文等级）。"""
    edu: list[dict] = []
    # 本科
    ug_start = date(grad_year - (4 if degree == "本科" else 7), 9, 1)
    ug_end = date(ug_start.year + 4, 6, rng.randint(20, 30))
    edu.append({
        "start_date": _fmt(ug_start),
        "end_date": _fmt(ug_end),
        "school": rng.choice(SCHOOLS),
        "college": rng.choice(["信息科学与工程学院", "计算机学院", "软件学院", "电子工程学院", "网络空间安全学院"]),
        "major": rng.choice(profile["majors"]),
        "education_level": "本科",
        "degree": "学士",
        "research_direction": "无",
        "lab_name": rng.choice(profile["labs"]) if rng.random() < 0.3 else None,
        "paper_level": "无",
        "is_current": False,
    })
    if degree in ("硕士", "博士"):
        pg_start = date(ug_end.year, 9, 1)
        is_current = grad_year >= APPLY_WINDOW_START.year
        pg_end = None if is_current else date(grad_year, 6, rng.randint(20, 30))
        level_label, deg_label = DEGREE_LEVEL_LABEL[degree]
        edu.append({
            "start_date": _fmt(pg_start),
            "end_date": _fmt(pg_end) if pg_end else None,
            "school": rng.choice(SCHOOLS),
            "college": rng.choice(["信息科学与工程学院", "计算机学院", "软件学院", "网络空间安全学院"]),
            "major": rng.choice(profile["majors"]),
            "education_level": level_label,
            "degree": deg_label,
            "research_direction": rng.choice(profile["research_directions"]),
            "lab_name": rng.choice(profile["labs"]),
            "paper_level": rng.choice(PAPER_LEVELS),
            "is_current": is_current,
        })
    return edu


def _build_skills(rng: random.Random, profile: dict) -> tuple[list[str], list[dict]]:
    """技能标签 + it_skill_items（与真实简历一致：含熟练度/时长/编程语言）。"""
    pool = profile["skills_pool"]
    core = profile["core_skills"]
    n = rng.randint(4, min(7, len(pool)))
    chosen = list(dict.fromkeys(core + rng.sample(pool, k=min(n, len(pool)))))
    rng.shuffle(chosen)
    langs = [s for s in chosen if s in ("Python", "Java", "Go", "C", "C++", "JavaScript", "TypeScript", "C#")]
    primary = langs[0] if langs else (chosen[0] if chosen else "Python")
    others = "、".join(langs[1:3]) if len(langs) > 1 else None
    it_items = [{
        "skill_name": chosen[0] if chosen else primary,
        "duration": rng.choice(["1年", "2年", "3年", "1年以下"]),
        "proficiency": rng.choice(["了解", "熟练", "精通"]),
        "primary_languages": primary,
        "other_languages": others,
    }]
    return chosen, it_items


def _build_internships(
    rng: random.Random,
    profile: dict,
    family: str,
    degree: str,
    grad_year: int,
    apply_time: date,
    skills: list[str],
) -> list[dict]:
    """构造实习经历（驱动 years_experience 估算）。部分简历无实习（真实分布）。"""
    templates = profile.get("internships") or []
    if not templates:
        return []
    # 应届/在读偏少实习经验；高学历/往届偏多
    max_n = 2 if degree in ("硕士", "博士") else 1
    n = rng.randint(0, max_n)
    if n == 0:
        return []
    spans: list[dict] = []
    used_companies: set[str] = set()
    # 实习写在投递日前，避免出现"投递后才结束的历史实习"。
    base = min(apply_time - timedelta(days=rng.randint(20, 120)), date(grad_year, 5, 31))
    chosen_templates = rng.sample(templates, k=min(n, len(templates)))
    for i, (dept, title, desc) in enumerate(chosen_templates):
        company, company_type = _real_company(rng, family, used_companies)
        start = base - timedelta(days=rng.randint(70, 360) * (i + 1))
        dur_days = rng.randint(90, 360)
        end = min(start + timedelta(days=dur_days), apply_time - timedelta(days=rng.randint(1, 25)))
        if end < start:
            end = start + timedelta(days=rng.randint(30, 90))
        is_current = rng.random() < 0.15 and i == 0
        spans.append({
            "start_date": _fmt(start),
            "end_date": None if is_current else _fmt(end),
            "company": company,
            "company_type": company_type,
            "department": dept,
            "title": title,
            "work_type": "实习",
            "description": _personalized_internship_desc(rng, family, desc, company, skills),
            "is_current": is_current,
        })
    return spans


def _build_projects(
    rng: random.Random,
    profile: dict,
    family: str,
    grad_year: int,
    apply_time: date,
    skills: list[str],
) -> list[dict]:
    templates = profile["projects"]
    n = min(len(templates), rng.randint(1, len(templates)))
    chosen = rng.sample(templates, k=n)
    out = []
    for name, desc, resp in chosen:
        latest_end = min(apply_time - timedelta(days=rng.randint(5, 80)), date(grad_year, 6, 30))
        start = latest_end - timedelta(days=rng.randint(120, 420))
        end = latest_end
        project_name, project_desc, project_resp = _personalized_project(rng, family, name, desc, resp, skills)
        out.append({
            "start_date": _fmt(start),
            "end_date": _fmt(end),
            "name": project_name,
            "description": project_desc,
            "responsibility": project_resp,
            "is_current": False,
        })
    return out


def _build_awards(rng: random.Random, profile: dict) -> list[dict]:
    templates = profile.get("awards") or []
    if not templates or rng.random() < 0.35:
        return []
    name, level, desc = rng.choice(templates)
    return [{"has_award": "是", "name": name, "level": level, "description": desc}]


def _section_text(doc: dict) -> dict:
    """拼装 section_text（与真实解析结构一致，用于 BM25 段落级检索）。"""
    def join_items(items, fields):
        parts = []
        for it in items:
            seg = " ".join(str(it.get(f)) for f in fields if it.get(f))
            if seg:
                parts.append(seg)
        return "\n".join(parts)

    cand = doc["candidate"]
    return {
        "personal_info": f"{cand['name']} {cand['highest_degree']} {cand['school']} {cand['major']}",
        "education": join_items(doc["education"], ["school", "college", "major", "degree", "research_direction", "lab_name"]),
        "internships": join_items(doc["internships"], ["department", "title", "description"]),
        "projects": join_items(doc["projects"], ["name", "description", "responsibility"]),
        "it_skills": " ".join(doc["skills"]),
        "languages": " ".join(str(v) for v in doc["languages"].values() if v),
        "awards": join_items(doc["awards"], ["name", "level", "description"]),
        "offer_internship": " ".join(str(v) for v in doc["offer_internship"].values() if v),
        "expected_work_city": " ".join(doc["application"]["expected_work_cities"]),
    }


def _build_one(rng: random.Random, family: str, profile: dict, idx: int, used_names: set[str]) -> dict:
    resume_id = f"M2026{idx:04d}"
    name = _unique_name(rng, used_names, idx)
    degree = rng.choice(DEGREE_CHOICES)
    grad_year = rng.choice([2024, 2025, 2026, 2026, 2027])
    position = rng.choice(profile["positions"])
    position_code = rng.choice(profile["position_codes"])
    apply_time = _rng_date(rng, APPLY_WINDOW_START, APPLY_WINDOW_END)
    city = rng.choice(CITIES)
    cities = [city] if rng.random() < 0.6 else rng.sample(CITIES, k=2)
    birth_year = grad_year - (22 if degree == "本科" else (25 if degree == "硕士" else 28)) - rng.randint(0, 2)

    education = _build_education(rng, profile, degree, grad_year)
    skills, it_items = _build_skills(rng, profile)
    internships = _build_internships(rng, profile, family, degree, grad_year, apply_time, skills)
    projects = _build_projects(rng, profile, family, grad_year, apply_time, skills)
    awards = _build_awards(rng, profile)
    can_intern = rng.random() < 0.5
    offer = {
        "post_graduation_intention": rng.choice([None, "全职", "继续深造"]),
        "can_intern": "是" if can_intern else "否",
        "available_start_date": _fmt(_rng_date(rng, apply_time, APPLY_WINDOW_END + timedelta(days=200))) if can_intern else None,
        "weekly_workdays": str(rng.choice([3, 4, 5])) if can_intern else None,
        "internship_period": rng.choice(INTERN_PERIODS) if can_intern else None,
    }

    doc: dict[str, Any] = {
        "resume_id": resume_id,
        "parse_status": "ok",
        "parse_errors": [],
        "parser_version": PARSER_VERSION,
        "file": {
            "name": f"{COMPANY}-{position}({position_code})-{name}({resume_id}).doc",
            "sha256": hashlib.sha256(resume_id.encode()).hexdigest(),
            "detected_type": "synthetic",
            "encoding": "utf-8",
        },
        "application": {
            "candidate_no": resume_id,
            "apply_time": _fmt(apply_time),
            "company": COMPANY,
            "position_code": position_code,
            "position_name": position,
            "wishes": [{"rank": 1, "position_name": position, "company": COMPANY}],
            "expected_work_cities": cities,
        },
        "candidate": {
            "name": name,
            "gender": rng.choice(["男", "女"]),
            "birth_date": f"{birth_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "current_city": rng.choice(CITIES),
            "highest_degree": degree,
            "graduation_date": f"{grad_year}-07-01",
            "school": education[-1]["school"],
            "major": education[-1]["major"],
            "phone": _phone(rng),
            "email": _email(rng, resume_id),
            # 不写 years_experience：交给 import_to_es 从实习估算
        },
        "education": education,
        "internships": internships,
        "projects": projects,
        "skills": skills,
        "it_skill_items": it_items,
        "languages": {
            "english_exam_score": rng.choice(ENGLISH_EXAM),
            "english_spoken_level": rng.choice(ENGLISH_SPOKEN),
        },
        "awards": awards,
        "offer_internship": offer,
        "_family": family,  # 调试用，写出前剔除
    }
    doc["section_text"] = _section_text(doc)
    return doc


def _weighted_plan(profiles: dict, total: int) -> list[str]:
    """按 weight 把 total 个名额分配到各族，返回 family 列表。"""
    weights = {k: v["weight"] for k, v in profiles.items()}
    wsum = sum(weights.values())
    plan: list[str] = []
    for fam, w in weights.items():
        plan += [fam] * round(total * w / wsum)
    # 校正到精确 total
    while len(plan) < total:
        plan.append(max(weights, key=weights.get))
    return plan[:total]


def generate(count: int = 200, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    # 约 88% 正常岗位族，约 12% hard negative
    n_hard = round(count * 0.12)
    n_normal = count - n_hard
    plan = _weighted_plan(POSITION_PROFILES, n_normal) + _weighted_plan(HARD_NEGATIVE_PROFILES, n_hard)
    rng.shuffle(plan)

    all_profiles = {**POSITION_PROFILES, **HARD_NEGATIVE_PROFILES}
    docs = []
    used_names: set[str] = set()
    for i, fam in enumerate(plan, start=1):
        docs.append(_build_one(rng, fam, all_profiles[fam], i, used_names))
    return docs


def _quality_stats(docs: list[dict]) -> dict[str, Any]:
    names = [doc["candidate"]["name"] for doc in docs]
    companies = [
        item["company"]
        for doc in docs
        for item in doc.get("internships") or []
        if item.get("company")
    ]
    project_signatures = [
        (
            item.get("name"),
            item.get("description"),
            item.get("responsibility"),
        )
        for doc in docs
        for item in doc.get("projects") or []
    ]
    internship_signatures = [
        (
            item.get("company"),
            item.get("department"),
            item.get("title"),
            item.get("description"),
        )
        for doc in docs
        for item in doc.get("internships") or []
    ]
    project_counts = Counter(project_signatures)
    internship_counts = Counter(internship_signatures)
    fake_companies = sorted({company for company in companies if "某" in company})
    return {
        "unique_names": len(set(names)),
        "total_names": len(names),
        "unique_companies": len(set(companies)),
        "total_internships": len(companies),
        "fake_companies": fake_companies,
        "duplicate_project_signatures": sum(n - 1 for n in project_counts.values() if n > 1),
        "duplicate_internship_signatures": sum(n - 1 for n in internship_counts.values() if n > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成贴合真实字段的模拟简历 JSONL")
    parser.add_argument("-o", "--output", default="data/ai_generated.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    docs = generate(args.count, args.seed)
    stats = _quality_stats(docs)
    assert stats["unique_names"] == stats["total_names"], "候选人姓名重复过多，请扩大姓名池或检查去重逻辑"
    assert not stats["fake_companies"], f"存在占位公司名: {stats['fake_companies']}"
    if stats["total_internships"]:
        min_unique_companies = min(20, stats["total_internships"])
        assert stats["unique_companies"] >= min_unique_companies, "真实公司池多样性不足"
    assert stats["duplicate_project_signatures"] == 0, "项目经历存在整段重复"

    # 自检 + 写出（剔除调试字段 _family）
    families: dict[str, int] = {}
    cov = {"languages": 0, "awards": 0, "offer_can_intern": 0, "internships": 0}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            families[doc["_family"]] = families.get(doc["_family"], 0) + 1
            if doc["languages"].get("english_exam_score"):
                cov["languages"] += 1
            if doc["awards"]:
                cov["awards"] += 1
            if doc["offer_internship"].get("can_intern") == "是":
                cov["offer_can_intern"] += 1
            if doc["internships"]:
                cov["internships"] += 1
            doc.pop("_family", None)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    ids = [d["resume_id"] for d in docs]
    assert len(ids) == len(set(ids)) == args.count, "resume_id 不唯一或数量不符"
    print(f"已写出 {len(docs)} 份到 {out_path}")
    print("岗位族分布：")
    for fam, n in sorted(families.items(), key=lambda kv: -kv[1]):
        tag = " [hard-neg]" if fam.startswith("hn_") else ""
        print(f"  {fam:20} {n:>3}{tag}")
    print("字段覆盖率：")
    for k, n in cov.items():
        print(f"  {k:20} {n}/{len(docs)} ({n/len(docs):.0%})")
    print("多样性自检：")
    print(f"  unique_names         {stats['unique_names']}/{stats['total_names']}")
    print(f"  unique_companies     {stats['unique_companies']}/{stats['total_internships']}")
    print(f"  duplicate_projects   {stats['duplicate_project_signatures']}")
    print(f"  duplicate_internships {stats['duplicate_internship_signatures']}")


if __name__ == "__main__":
    main()
