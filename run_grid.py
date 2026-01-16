#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank Eval 그리드 서치 실행기 (함수별 설명 강화 버전)

실행 흐름(한 줄 요약)
- grid.yaml의 파라미터 조합을 전부 돌면서
- template.json을 _render/template로 렌더링해 실제 쿼리를 만들고
- /{index}/_rank_eval로 metric 점수를 수집해서
- best.json(최고 조합) + best_debug.json(최고 조합 top-k 검색결과 vs gold)을 저장한다.

PowerShell 실행:
  python -u .\run_grid.py --config .\config.yaml
"""

import argparse
import itertools
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import logging
import urllib3

import pandas as pd
import requests
import yaml
from tqdm import tqdm

# 자가서명 인증서 + verify=False 사용 시 뜨는 경고 숨김
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



### 함수 별 기능 요약 ###
# load_json / load_yaml : 입력 파일 읽기

# load_template_source : template.json에서 source만 꺼내기

# setup_logging : 콘솔+파일 로그 설정

# to_py : pandas/numpy 타입을 JSON 저장 가능한 타입으로 변환

# post_json : ES에 POST 요청(재시도/에러바디 출력/verify=False/auth)

# iter_combinations : grid.yaml 조합 만들기

# render_template : _render/template로 mustache 렌더링

# build_rank_eval_body : _rank_eval 요청 바디 만들기(+ ratings에 _index 채우기)

# fetch_top_hits : best 조합으로 _search해서 top-k 결과 가져오기

# main : 전체 실행 흐름(그리드 실행 → 저장 → best → best_debug)



# -----------------------------
# 파일 로드/저장 관련 유틸
# -----------------------------
def load_json(path: str) -> Any:
    """
    JSON 파일을 파이썬 객체로 로드한다.
    예: queries.json, gold.json, template.json 등
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Any:
    """
    YAML 파일을 파이썬 객체로 로드한다.
    예: config.yaml, grid.yaml
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_template_source(path: str = "template.json") -> Any:
    """
    template.json의 구조는 보통 이런 형태:
      { "source": { ... mustache 템플릿 ... } }

    run_grid.py에서는 'source' 내부만 쓰기 때문에
    template.json에서 source만 꺼내 반환한다.
    """
    obj = load_json(path)
    return obj["source"]


def ensure_dir(path: str) -> None:
    """폴더가 없으면 만든다."""
    os.makedirs(path, exist_ok=True)


def setup_logging(output_dir: str, level: str = "INFO") -> str:
    """
    콘솔 + 파일(outputs/run.log)에 동시에 로그를 남기도록 설정한다.
    """
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "run.log")
    lvl = getattr(logging, str(level).upper(), logging.INFO)

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


# -----------------------------
# JSON 직렬화 오류 방지 유틸
# -----------------------------
def to_py(obj: Any) -> Any:
    """
    pandas/numpy 타입(int64/float64 등)이 json.dump에서 터지는 문제 방지.

    - numpy.int64 -> int
    - numpy.float64 -> float
    - dict/list 내부도 재귀적으로 변환
    """
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_py(v) for v in obj]
    return obj


def fmt_secs(seconds: float) -> str:
    """초(second)를 사람이 읽기 쉬운 형태(s / m / h)로 변환."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


# -----------------------------
# HTTP 호출(재시도/백오프/에러 출력)
# -----------------------------
def post_json(
    url: str,
    body: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff_base: float,
    auth: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    Elasticsearch에 POST(JSON)를 보내는 공통 함수.

    주요 기능
    - verify=False : 사내/자가서명 인증서 환경에서 SSL 검증을 끔
    - auth : Basic Auth (username, password)
    - retries/backoff : 네트워크/ES 순간 장애 시 재시도
    - 4xx/5xx면 응답 바디(JSON)를 그대로 출력해서 디버깅이 빠르게 함
    """
    last_exc = None

    for i in range(retries):
        try:
            r = requests.post(url, json=body, timeout=timeout, verify=False, auth=auth)

            if r.status_code >= 400:
                # 에러 바디 출력: ES가 왜 싫어하는지 바로 보려고
                try:
                    tqdm.write(
                        f"[HTTP {r.status_code}] {url}\n"
                        f"{json.dumps(r.json(), ensure_ascii=False, indent=2)}"
                    )
                except Exception:
                    tqdm.write(f"[HTTP {r.status_code}] {url}\n{r.text}")
                r.raise_for_status()

            return r.json()

        except Exception as e:
            last_exc = e
            time.sleep(backoff_base**i)

    raise last_exc


# -----------------------------
# grid / template / rank_eval 구성 로직
# -----------------------------
def iter_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    grid.yaml의 각 파라미터 리스트를 받아 모든 조합을 만든다.

    예)
      boost_title: [0.5, 1.0]
      operator: ["or", "and"]

    -> 4개 조합
      {boost_title:0.5, operator:"or"}
      {boost_title:0.5, operator:"and"}
      {boost_title:1.0, operator:"or"}
      {boost_title:1.0, operator:"and"}
    """
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def render_template(
    es_url: str,
    template_source: Any,
    params: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff_base: float,
    auth: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    Mustache 템플릿(template.json)을 실제 Query DSL로 변환(render)한다.

    호출:
      POST {es_url}/_render/template
    body:
      { "source": template_source, "params": params }

    반환:
      response["template_output"] (실제 검색 요청 body 형태)
    """
    body = {"source": template_source, "params": params}
    out = post_json(
        f"{es_url}/_render/template",
        body,
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        auth=auth,
    )
    if "template_output" not in out:
        raise RuntimeError(f"Unexpected render response keys={list(out.keys())}")
    return out["template_output"]


def build_rank_eval_body(
    rendered_by_qid: Dict[str, Dict[str, Any]],
    queries: List[Dict[str, Any]],
    gold: Dict[str, List[Dict[str, Any]]],
    metric_type: str,
    metric_params: Dict[str, Any],
    index: str,
) -> Dict[str, Any]:
    """
    /_rank_eval 요청 바디를 만든다.

    중요 포인트
    - ES rank_eval의 requests는 '배열' 형태가 가장 호환성이 좋음.
    - ratings는 반드시 _index가 필요할 수 있음(버전/설정에 따라).
      그래서 gold.json에 _index가 없으면 자동으로 index를 채워 넣는다.

    최종 구조:
    {
      "requests": [
        {"id": "q1", "request": {...}, "ratings": [{"_index": "...", "_id":"doc1", "rating":3}, ...]},
        ...
      ],
      "metric": { "precision": { "k": 10, ... } }
    }
    """
    req_list = []

    for q in queries:
        qid = q["id"]
        if qid not in rendered_by_qid:
            raise KeyError(f"Missing rendered query for {qid}")
        if qid not in gold:
            raise KeyError(f"Missing gold ratings for {qid}")

        rated = []
        for d in gold[qid]:
            dd = dict(d)
            dd.setdefault("_index", index)
            rated.append(dd)

        req_list.append(
            {
                "id": qid,
                "request": rendered_by_qid[qid],
                "ratings": rated,
            }
        )

    return {"requests": req_list, "metric": {metric_type: metric_params}}


# -----------------------------
# best_debug 용: 실제 top-k 결과 가져오기
# -----------------------------
def fetch_top_hits(
    es_url: str,
    index: str,
    rendered_query: Dict[str, Any],
    size: int,
    timeout: int,
    retries: int,
    backoff_base: float,
    auth: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    best 조합(최고 파라미터)로 실제 검색 결과를 보고 싶을 때 사용.

    - rendered_query를 그대로 /{index}/_search에 넣어 top-k를 가져오고
    - gold와 나란히 놓고 "눈으로 비교"할 수 있게 출력 형태를 정리한다.
    """
    body = dict(rendered_query)
    body["size"] = size
    body.setdefault("_source", ["title", "content"])

    resp = post_json(
        f"{es_url}/{index}/_search",
        body,
        timeout=timeout,
        retries=retries,
        backoff_base=backoff_base,
        auth=auth,
    )

    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source") or {}
        out.append(
            {
                "_id": h.get("_id"),
                "_score": h.get("_score"),
                "title": src.get("title"),
                "content": src.get("content"),
            }
        )
    return out


# -----------------------------
# main: 전체 실행 흐름
# -----------------------------
def main():
    # 0) 실행 인자
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    # 1) config.yaml 읽기
    cfg = load_yaml(args.config)

    es_url = str(cfg["es_url"]).rstrip("/")
    index = str(cfg["index"])

    timeout = int(cfg.get("timeout_sec", 30))
    retries = int(cfg.get("retries", 3))
    backoff = float(cfg.get("backoff_base", 1.5))

    main_metric = str(cfg.get("main_metric", "precision"))
    max_combos = int(cfg.get("max_combinations", 0))
    shuffle = bool(cfg.get("shuffle_combinations", False))

    output_dir = str(cfg.get("output_dir", "outputs"))
    log_path = setup_logging(output_dir, level=str(cfg.get("log_level", "INFO")))

    # 2) Basic Auth(선택)
    auth = None
    auth_cfg = cfg.get("auth") or {}
    if auth_cfg.get("username") and auth_cfg.get("password"):
        auth = (str(auth_cfg["username"]), str(auth_cfg["password"]))

    logging.info("=== RankEval Grid Search Start ===")
    logging.info(f"es_url={es_url}, index={index}")
    logging.info(f"timeout={timeout}s retries={retries} backoff={backoff}")
    logging.info(f"main_metric={main_metric} shuffle={shuffle} max_combinations={max_combos}")
    logging.info(f"output_dir={output_dir} log={log_path}")
    logging.info(f"auth={'enabled' if auth else 'disabled'}")

    # 3) 입력 파일 로드
    template_source = load_template_source("template.json")
    queries = load_json("queries.json")      # qid + (searchKeyword 등) query별 파라미터
    gold = load_json("gold.json")            # qid별 정답 문서 목록(_id, rating)
    grid = load_yaml("grid.yaml")            # 실험할 파라미터 조합
    metrics_cfg = cfg.get("metrics", {})     # precision/recall/mrr/dcg 등

    # 4) 그리드 조합 생성
    combos = iter_combinations(grid)
    if shuffle:
        random.shuffle(combos)
    if max_combos > 0:
        combos = combos[:max_combos]

    if not combos:
        logging.warning("grid 조합이 0개라서 종료합니다.")
        return
    if not metrics_cfg:
        logging.warning("metrics가 비어있어 rank_eval 호출 없이 render만 합니다.")

    # metric들의 k 중 가장 큰 값으로 size를 맞춤
    max_k = max(int(m.get("k", 10)) for m in metrics_cfg.values()) if metrics_cfg else 10
    logging.info(f"queries={len(queries)}, combos={len(combos)}, metrics={list(metrics_cfg.keys())}, max_k={max_k}")

    # 5) 그리드 실행
    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    combos_pbar = tqdm(combos, desc="Grid combos", unit="combo", dynamic_ncols=True)

    for ci, combo in enumerate(combos_pbar, start=1):
        combo_t0 = time.time()
        combos_pbar.set_postfix_str(f"{ci}/{len(combos)}")

        combo_result = {
            "params": combo,   # 이번 조합 파라미터
            "scores": {},      # metric별 전체 점수
            "errors": [],      # render/rank_eval 오류 기록
        }

        # (A) qid별 렌더링 결과(실제 Query DSL)를 만든다
        rendered_by_qid: Dict[str, Dict[str, Any]] = {}

        for q in tqdm(queries, desc=f"Render (combo {ci})", leave=False, unit="q", dynamic_ncols=True):
            qid = q["id"]

            # 쿼리별 파라미터 + 그리드 조합 파라미터 합치기
            params: Dict[str, Any] = {}
            params.update(q.get("params", {}))
            params.update(combo)

            try:
                rendered = render_template(es_url, template_source, params, timeout, retries, backoff, auth=auth)
                rendered = dict(rendered)
                rendered["size"] = max_k
                rendered_by_qid[qid] = rendered
            except Exception as e:
                combo_result["errors"].append({"stage": "render", "query_id": qid, "error": str(e)})
                tqdm.write(f"[WARN] render failed qid={qid}: {e}")

        # 렌더링 하나라도 실패하면 rank_eval 스킵(원래 로직 유지)
        if len(rendered_by_qid) != len(queries):
            all_results.append(combo_result)
            continue

        # (B) metric별로 /_rank_eval 호출
        for metric_type, metric_params in tqdm(list(metrics_cfg.items()), desc=f"RankEval (combo {ci})",
                                               leave=False, unit="metric", dynamic_ncols=True):
            try:
                body = build_rank_eval_body(rendered_by_qid, queries, gold, metric_type, metric_params, index)
                resp = post_json(
                    f"{es_url}/{index}/_rank_eval",
                    body,
                    timeout=timeout,
                    retries=retries,
                    backoff_base=backoff,
                    auth=auth,
                )
                combo_result["scores"][metric_type] = resp.get("metric_score")
            except Exception as e:
                combo_result["errors"].append({"stage": "rank_eval", "metric": metric_type, "error": str(e)})
                tqdm.write(f"[WARN] rank_eval failed metric={metric_type}: {e}")

        all_results.append(combo_result)

        # 진행 로그(대충 얼마나 남았나)
        combo_elapsed = time.time() - combo_t0
        total_elapsed = time.time() - t0
        avg_per_combo = total_elapsed / ci
        remaining = avg_per_combo * (len(combos) - ci)
        tqdm.write(
            f"[{ci}/{len(combos)}] done scores={combo_result['scores']} "
            f"errors={len(combo_result['errors'])} combo_elapsed={fmt_secs(combo_elapsed)} ETA={fmt_secs(remaining)}"
        )

    # 6) results.json 저장(전체 raw)
    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(to_py(all_results), f, ensure_ascii=False, indent=2)

    # 7) results.csv 저장(분석/정렬용)
    rows = []
    for r in all_results:
        row = {}
        row.update(r.get("params", {}))
        row.update(r.get("scores", {}))
        row["error_count"] = len(r.get("errors", []))
        rows.append(row)

    df = pd.DataFrame(rows)
    results_csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")

    # 8) best 선정(동점 tie-break: dcg -> recall -> precision)
    best = None
    if len(df) > 0:
        df_ok = df[df["error_count"] == 0].copy()
        if len(df_ok) == 0:
            df_ok = df.copy()

        sort_cols: List[str] = []
        sort_asc: List[bool] = []

        if main_metric in df_ok.columns:
            sort_cols.append(main_metric); sort_asc.append(False)

        for c in ["dcg", "recall", "precision"]:
            if c in df_ok.columns and c not in sort_cols:
                sort_cols.append(c); sort_asc.append(False)

        if "error_count" in df_ok.columns:
            sort_cols.append("error_count"); sort_asc.append(True)

        if sort_cols:
            df_ok = df_ok.sort_values(by=sort_cols, ascending=sort_asc, na_position="last")

        if len(df_ok) > 0:
            best_params = {k: df_ok.iloc[0][k] for k in grid.keys() if k in df_ok.columns}
            best = {
                "main_metric": main_metric,
                "tie_breaks": ["dcg", "recall", "precision"],
                "best_params": best_params,
                "best_scores": {
                    m: (float(df_ok.iloc[0][m]) if m in df_ok.columns and pd.notna(df_ok.iloc[0][m]) else None)
                    for m in metrics_cfg.keys()
                },
            }

    best_json_path = os.path.join(output_dir, "best.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(to_py(best), f, ensure_ascii=False, indent=2)

    # 9) best_debug.json 생성(최고 조합으로 qid별 top-k 검색 결과를 gold와 함께 저장)
    if best and best.get("best_params"):
        debug_k = int(cfg.get("debug_k", 10))
        best_params = to_py(best["best_params"])

        per_query_debug: Dict[str, Any] = {}

        for q in queries:
            qid = q["id"]
            q_params = {}
            q_params.update(q.get("params", {}))
            q_params.update(best_params)

            try:
                rendered = render_template(es_url, template_source, q_params, timeout, retries, backoff, auth=auth)
                hits = fetch_top_hits(es_url, index, rendered, debug_k, timeout, retries, backoff, auth=auth)
                per_query_debug[qid] = {
                    "query_params": q.get("params", {}),
                    "gold": gold.get(qid, []),
                    "hits": hits,
                }
            except Exception as e:
                per_query_debug[qid] = {
                    "query_params": q.get("params", {}),
                    "gold": gold.get(qid, []),
                    "error": str(e),
                }

        best_debug = {
            "index": index,
            "main_metric": best.get("main_metric"),
            "tie_breaks": best.get("tie_breaks"),
            "best_params": best.get("best_params"),
            "best_scores": best.get("best_scores"),
            "debug_k": debug_k,
            "per_query": per_query_debug,
        }

        best_debug_path = os.path.join(output_dir, "best_debug.json")
        with open(best_debug_path, "w", encoding="utf-8") as f:
            json.dump(to_py(best_debug), f, ensure_ascii=False, indent=2)

        logging.info(f"Saved: {best_debug_path}")
        tqdm.write(f"Saved: {best_debug_path}")

    # 10) 종료 로그
    total_elapsed = time.time() - t0
    logging.info("=== Done ===")
    logging.info(f"Saved: {results_json_path}")
    logging.info(f"Saved: {results_csv_path}")
    logging.info(f"Saved: {best_json_path}")
    logging.info(f"Total elapsed: {fmt_secs(total_elapsed)}")

    tqdm.write(f"Saved: {results_json_path}")
    tqdm.write(f"Saved: {results_csv_path}")
    tqdm.write(f"Saved: {best_json_path}")
    tqdm.write(f"Total elapsed: {fmt_secs(total_elapsed)}")
    tqdm.write(f"Log: {log_path}")


if __name__ == "__main__":
    main()
