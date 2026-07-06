"""导入命令行入口：解析参数并调用 import_resumes。"""
from __future__ import annotations

import argparse
import json

from indexing.mappings import (
    DEFAULT_ALIAS,
    DEFAULT_ES_URL,
    DEFAULT_EVIDENCE_ALIAS,
    DEFAULT_EVIDENCE_INDEX,
    DEFAULT_INDEX,
)
from indexing.pipeline import import_resumes


def main() -> int:
    parser = argparse.ArgumentParser(description="Import parsed resumes into Elasticsearch.")
    parser.add_argument("data_path", nargs="?", default="data")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--evidence-index", default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--evidence-alias", default=DEFAULT_EVIDENCE_ALIAS)
    parser.add_argument("--no-recreate", action="store_true")
    parser.add_argument(
        "--delete-missing",
        action="store_true",
        help="When importing into an existing index, delete documents not present in this import set.",
    )
    args = parser.parse_args()

    result = import_resumes(
        data_path=args.data_path,
        es_url=args.es_url,
        index=args.index,
        alias=args.alias,
        evidence_index=args.evidence_index,
        evidence_alias=args.evidence_alias,
        recreate=not args.no_recreate,
        delete_missing=args.delete_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
