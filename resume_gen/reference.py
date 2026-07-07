"""真实世界参照数据池：院校、公司、城市、姓名、英语水平等。

本模块只提供"现实中真实存在"的取值池与确定性采样工具，不含任何人物画像逻辑。
院校直接读 school_tiers.json（与检索侧院校档位共用同一份真实名单），保证生成的
毕业院校都能在现实里对上号，也天然覆盖长尾、支撑"千人千面"。
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHOOL_TIERS_PATH = PROJECT_ROOT / "school_tiers.json"


# --- 院校 -----------------------------------------------------------------
@lru_cache(maxsize=1)
def _school_tiers() -> dict[str, list[str]]:
    data = json.loads(SCHOOL_TIERS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, list)}


@lru_cache(maxsize=1)
def _weighted_school_pool() -> list[tuple[str, int]]:
    """(校名, 权重) 列表。名校权重高（更多候选人出自名校，符合大厂投递分布），
    但长尾院校全部可达，保证院校维度足够分散、避免所有人都来自清北。

    只保留中文校名（过滤 school_tiers 里为匹配英文写法而并列的拉丁字母条目）。
    """
    tiers = _school_tiers()
    t985 = set(tiers.get("985", []))
    t211 = set(tiers.get("211", []))
    shuang = set(tiers.get("双一流", []))
    pool: dict[str, int] = {}
    for name in shuang | t211 | t985:
        if not _is_chinese_name(name):
            continue
        if name in t985:
            weight = 5
        elif name in t211:
            weight = 3
        else:
            weight = 2
        pool[name] = weight
    return sorted(pool.items())


# 少量海外名校，给约 5% 的候选人一个"海外背景"，进一步拉开画像差异。
OVERSEAS_SCHOOLS = [
    "新加坡国立大学", "南洋理工大学", "香港大学", "香港科技大学", "香港中文大学",
    "帝国理工学院", "爱丁堡大学", "曼彻斯特大学", "多伦多大学", "墨尔本大学",
    "南加州大学", "卡内基梅隆大学", "东京大学",
]


def _is_chinese_name(name: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in name)


def sample_school(rng: random.Random, *, allow_overseas: bool = True) -> str:
    if allow_overseas and rng.random() < 0.05:
        return rng.choice(OVERSEAS_SCHOOLS)
    names, weights = zip(*_weighted_school_pool())
    return rng.choices(names, weights=weights, k=1)[0]


def sample_school_progression(rng: random.Random) -> tuple[str, str]:
    """返回 (本科院校, 研究生院校)。约 35% 的人研究生留在本校，其余换校；
    海外本科很少，海外研究生略多（出国读研）——符合真实升学路径。"""
    undergrad = sample_school(rng, allow_overseas=False)
    if rng.random() < 0.35:
        return undergrad, undergrad
    grad = sample_school(rng, allow_overseas=True)
    return undergrad, grad


# --- 城市 -----------------------------------------------------------------
# 一线 + 新一线，覆盖投递热点城市。
TIER1_CITIES = ["北京", "上海", "深圳", "广州"]
NEW_TIER1_CITIES = ["杭州", "成都", "武汉", "南京", "西安", "苏州", "重庆", "长沙", "合肥", "天津"]
ALL_CITIES = TIER1_CITIES + NEW_TIER1_CITIES


def sample_city(rng: random.Random) -> str:
    # 一线城市投递集中，给更高权重。
    weights = [4] * len(TIER1_CITIES) + [2] * len(NEW_TIER1_CITIES)
    return rng.choices(ALL_CITIES, weights=weights, k=1)[0]


def sample_expected_cities(rng: random.Random, current_city: str) -> list[str]:
    """期望工作城市：多数人 1 个（常与当前城市一致），部分人 2 个。"""
    if rng.random() < 0.6:
        return [current_city]
    second = current_city
    while second == current_city:
        second = sample_city(rng)
    return [current_city, second]


# --- 姓名 -----------------------------------------------------------------
FAMILY_NAMES = list(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁"
    "潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱侯江尹薛"
)
# 偏男性用字、偏女性用字、中性用字——用于让名字与性别弱相关，读起来更自然。
GIVEN_MALE = [
    "伟", "强", "磊", "军", "勇", "杰", "涛", "斌", "波", "鹏", "宇", "浩", "航", "凯",
    "帆", "刚", "锋", "毅", "晨", "轩", "昊", "泽", "宇轩", "浩然", "子轩", "俊杰", "宇航",
    "文博", "嘉豪", "泽宇", "明轩", "志强", "建国", "国栋", "天翔", "博文", "承宇",
]
GIVEN_FEMALE = [
    "芳", "娜", "敏", "静", "丽", "艳", "娟", "霞", "婷", "颖", "雪", "琳", "倩", "洁",
    "璐", "欣", "怡", "妍", "婧", "梦", "欣怡", "雨欣", "梓涵", "思颖", "佳怡", "梦琪",
    "雅婷", "诗涵", "语嫣", "婉婷", "晓彤", "馨月", "紫萱", "若曦",
]
GIVEN_NEUTRAL = [
    "宁", "乐", "晗", "然", "睿", "越", "洋", "阳", "佳", "鑫", "越洋", "一诺", "子墨",
    "沐辰", "亦然", "书航", "云帆",
]


def sample_name(rng: random.Random, gender: str) -> str:
    family = rng.choice(FAMILY_NAMES)
    if gender == "男":
        given_pool = GIVEN_MALE + GIVEN_NEUTRAL
    else:
        given_pool = GIVEN_FEMALE + GIVEN_NEUTRAL
    return family + rng.choice(given_pool)


def dedup_name(rng: random.Random, base: str, used: set[str], gender: str) -> str:
    """避免小样本抽样撞名：撞名时换一个 given，仍保持真实中文名的样子。"""
    if base not in used:
        used.add(base)
        return base
    for _ in range(40):
        candidate = sample_name(rng, gender)
        if candidate not in used:
            used.add(candidate)
            return candidate
    # 极端兜底：家姓 + 双字名
    family = base[0]
    for _ in range(40):
        given = rng.choice(GIVEN_MALE + GIVEN_FEMALE + GIVEN_NEUTRAL)
        candidate = family + given
        if candidate not in used:
            used.add(candidate)
            return candidate
    used.add(base)
    return base


# --- 英语水平 --------------------------------------------------------------
ENGLISH_EXAM_SCORES = [
    "CET 4: 425", "CET 4: 480", "CET 4: 512", "CET 4: 560",
    "CET 6: 426", "CET 6: 468", "CET 6: 502", "CET 6: 540", "CET 6: 580",
    "雅思 6.0", "雅思 6.5", "雅思 7.0", "托福 90", "托福 100", "TEM 8",
]
ENGLISH_SPOKEN_LEVELS = ["简单会话", "可面试", "良好", "熟练", "流利"]


def sample_english(rng: random.Random) -> tuple[str | None, str | None]:
    # 少数简历不填英语成绩（真实分布里也有空缺）。
    if rng.random() < 0.12:
        return None, None
    return rng.choice(ENGLISH_EXAM_SCORES), rng.choice(ENGLISH_SPOKEN_LEVELS)


# --- 联系方式 --------------------------------------------------------------
EMAIL_DOMAINS = ["126.com", "163.com", "qq.com", "gmail.com", "foxmail.com", "outlook.com", "139.com"]


def sample_phone(rng: random.Random) -> str:
    prefix = rng.choice(["13", "15", "17", "18", "19", "136", "159", "188", "186", "133"])
    remaining = 11 - len(prefix)
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(remaining))


def sample_email(rng: random.Random, name_pinyin_seed: str) -> str:
    import hashlib

    style = rng.random()
    digest = hashlib.md5(name_pinyin_seed.encode()).hexdigest()
    if style < 0.4:
        handle = digest[:8]
    elif style < 0.7:
        handle = digest[:6] + str(rng.randint(1970, 2003))
    else:
        handle = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 9)))
    return f"{handle}@{rng.choice(EMAIL_DOMAINS)}"


# --- 企业名录（真实公司，按岗位族倾斜）------------------------------------
# company_type 保持在真实解析可接受的枚举内。
DEFAULT_COMPANIES: list[tuple[str, str]] = [
    ("腾讯", "私营/民营企业"), ("阿里巴巴", "私营/民营企业"), ("百度", "私营/民营企业"),
    ("京东", "私营/民营企业"), ("美团", "私营/民营企业"), ("华为", "私营/民营企业"),
    ("字节跳动", "私营/民营企业"), ("小米集团", "私营/民营企业"), ("网易", "私营/民营企业"),
]

COMPANIES_BY_FAMILY: dict[str, list[tuple[str, str]]] = {
    "security_research": [
        ("360数字安全集团", "私营/民营企业"), ("奇安信集团", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"), ("绿盟科技", "私营/民营企业"),
        ("启明星辰", "私营/民营企业"), ("天融信", "私营/民营企业"),
        ("安恒信息", "私营/民营企业"), ("亚信安全", "私营/民营企业"),
        ("知道创宇", "私营/民营企业"), ("腾讯", "私营/民营企业"),
    ],
    "blue_team": [
        ("奇安信集团", "私营/民营企业"), ("360数字安全集团", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"), ("绿盟科技", "私营/民营企业"),
        ("启明星辰", "私营/民营企业"), ("安恒信息", "私营/民营企业"),
        ("亚信安全", "私营/民营企业"), ("腾讯", "私营/民营企业"),
        ("阿里云", "私营/民营企业"), ("中国工商银行", "国有企业"),
    ],
    "red_team": [
        ("360数字安全集团", "私营/民营企业"), ("奇安信集团", "私营/民营企业"),
        ("绿盟科技", "私营/民营企业"), ("深信服科技", "私营/民营企业"),
        ("安恒信息", "私营/民营企业"), ("启明星辰", "私营/民营企业"),
        ("天融信", "私营/民营企业"), ("腾讯", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"), ("默安科技", "私营/民营企业"),
    ],
    "devsecops": [
        ("阿里云", "私营/民营企业"), ("腾讯云", "私营/民营企业"), ("华为云", "私营/民营企业"),
        ("京东科技", "私营/民营企业"), ("蚂蚁集团", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"), ("美团", "私营/民营企业"),
        ("快手", "私营/民营企业"), ("深信服科技", "私营/民营企业"),
    ],
    "ml_llm": [
        ("百度", "私营/民营企业"), ("阿里巴巴", "私营/民营企业"), ("腾讯", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"), ("科大讯飞", "私营/民营企业"),
        ("商汤科技", "私营/民营企业"), ("旷视科技", "私营/民营企业"),
        ("小红书", "私营/民营企业"), ("美团", "私营/民营企业"), ("智谱华章", "私营/民营企业"),
    ],
    "backend_go": [
        ("字节跳动", "私营/民营企业"), ("腾讯", "私营/民营企业"), ("快手", "私营/民营企业"),
        ("京东", "私营/民营企业"), ("美团", "私营/民营企业"), ("百度", "私营/民营企业"),
        ("哔哩哔哩", "私营/民营企业"), ("小米集团", "私营/民营企业"), ("滴滴出行", "私营/民营企业"),
    ],
    "backend_java": [
        ("阿里巴巴", "私营/民营企业"), ("蚂蚁集团", "私营/民营企业"), ("京东", "私营/民营企业"),
        ("美团", "私营/民营企业"), ("招商银行", "国有企业"), ("平安科技", "私营/民营企业"),
        ("携程", "私营/民营企业"), ("贝壳找房", "私营/民营企业"), ("用友网络", "私营/民营企业"),
    ],
    "backend_cpp": [
        ("华为", "私营/民营企业"), ("中兴通讯", "私营/民营企业"), ("腾讯", "私营/民营企业"),
        ("百度", "私营/民营企业"), ("字节跳动", "私营/民营企业"), ("商汤科技", "私营/民营企业"),
        ("海康威视", "私营/民营企业"), ("大华股份", "私营/民营企业"), ("寒武纪", "私营/民营企业"),
    ],
    "frontend": [
        ("字节跳动", "私营/民营企业"), ("腾讯", "私营/民营企业"), ("阿里巴巴", "私营/民营企业"),
        ("百度", "私营/民营企业"), ("美团", "私营/民营企业"), ("京东", "私营/民营企业"),
        ("小红书", "私营/民营企业"), ("哔哩哔哩", "私营/民营企业"), ("金山办公", "私营/民营企业"),
    ],
    "data_analysis": [
        ("美团", "私营/民营企业"), ("京东", "私营/民营企业"), ("阿里巴巴", "私营/民营企业"),
        ("字节跳动", "私营/民营企业"), ("腾讯", "私营/民营企业"), ("拼多多", "私营/民营企业"),
        ("滴滴出行", "私营/民营企业"), ("小红书", "私营/民营企业"), ("快手", "私营/民营企业"),
    ],
    "testing": [
        ("华为", "私营/民营企业"), ("腾讯", "私营/民营企业"), ("阿里巴巴", "私营/民营企业"),
        ("百度", "私营/民营企业"), ("京东", "私营/民营企业"), ("小米集团", "私营/民营企业"),
        ("中兴通讯", "私营/民营企业"), ("金山办公", "私营/民营企业"), ("携程", "私营/民营企业"),
    ],
    "product": [
        ("腾讯", "私营/民营企业"), ("阿里云", "私营/民营企业"), ("华为云", "私营/民营企业"),
        ("深信服科技", "私营/民营企业"), ("奇安信集团", "私营/民营企业"),
        ("绿盟科技", "私营/民营企业"), ("用友网络", "私营/民营企业"),
        ("金山办公", "私营/民营企业"), ("字节跳动", "私营/民营企业"),
    ],
    "hn_search_ops": [
        ("百度", "私营/民营企业"), ("阿里云", "私营/民营企业"), ("腾讯云", "私营/民营企业"),
        ("京东科技", "私营/民营企业"), ("华为云", "私营/民营企业"),
    ],
    "hn_it_ops": [
        ("中国移动", "国有企业"), ("中国电信", "国有企业"), ("国家电网", "国有企业"),
        ("招商银行", "国有企业"), ("中信银行", "国有企业"), ("中国联通", "国有企业"),
    ],
    "hn_cpp_biz": [
        ("用友网络", "私营/民营企业"), ("金蝶软件", "私营/民营企业"), ("广联达", "私营/民营企业"),
        ("航天信息", "国有企业"), ("浪潮软件", "国有企业"),
    ],
    "hn_bi_report": [
        ("帆软软件", "私营/民营企业"), ("用友网络", "私营/民营企业"), ("金蝶软件", "私营/民营企业"),
        ("京东", "私营/民营企业"), ("美团", "私营/民营企业"),
    ],
}


def company_catalog(family: str) -> list[tuple[str, str]]:
    return COMPANIES_BY_FAMILY.get(family, DEFAULT_COMPANIES)
