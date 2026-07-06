"""FastAPI 应用与 HTTP 路由。

这是最外层的 interface adapter：只做请求参数绑定、调用 service、返回响应，不含
检索业务逻辑。业务在 services 层，数据访问在 infrastructure 层。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from resume_search.config import INDEX_ALIAS, WEB_DIR
from resume_search.infrastructure.es_client import es_request as _es
from resume_search.services.search import search as _search

app = FastAPI(title="Resume Search Prototype")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/search")
def search(
    q: str = "",
    degree: str = "",
    cities: list[str] = Query(default=[]),
    skills: list[str] = Query(default=[]),
    min_years: float = 0,
    school_tiers: list[str] = Query(default=[]),
    limit: int = 0,
    offset: int = 0,
) -> dict[str, Any]:
    return _search(
        q=q,
        degree=degree,
        cities=cities,
        skills=skills,
        min_years=min_years,
        school_tiers=school_tiers,
        limit=limit,
        offset=offset,
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        result = _es("GET", "/_cluster/health")
        return {
            "es_online": True,
            "status": result.get("status", "unknown"),
            "indices": result.get("number_of_indices", 0),
        }
    except Exception:
        return {"es_online": False, "status": "offline", "indices": 0}


@app.get("/api/resumes/{resume_id}")
def get_resume(resume_id: str) -> dict[str, Any]:
    result = _es("GET", f"/{INDEX_ALIAS}/_doc/{resume_id}")
    return result.get("_source", {})
