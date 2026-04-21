from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AMAZON_RATINGS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/ratings_Amazon_Fashion.csv"
AMAZON_META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Amazon_Fashion.json.gz"
HM_DATASET = "dinhlnd1610/HM-Personalized-Fashion-Recommendations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public dataset samples and source manifests.")
    parser.add_argument("--output-dir", default="data/external", help="Directory for downloaded samples.")
    parser.add_argument("--hm-transactions", type=int, default=500, help="Number of H&M transactions to fetch.")
    return parser.parse_args()


def http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    urllib.request.urlretrieve(url, destination)


def fetch_rows(config: str, split: str, offset: int, length: int) -> List[dict]:
    params = {
        "dataset": HM_DATASET,
        "config": config,
        "split": split,
        "offset": offset,
        "length": min(length, 100),
    }
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params)
    payload = http_json(url)
    return [row["row"] for row in payload["rows"]]


def fetch_filter_rows(config: str, split: str, where: str) -> List[dict]:
    params = {
        "dataset": HM_DATASET,
        "config": config,
        "split": split,
        "where": where,
        "offset": 0,
        "length": 100,
    }
    url = "https://datasets-server.huggingface.co/filter?" + urllib.parse.urlencode(params)
    payload = http_json(url)
    return [row["row"] for row in payload["rows"]]


def paginated_rows(config: str, total_rows: int) -> pd.DataFrame:
    rows: List[dict] = []
    for offset in range(0, total_rows, 100):
        rows.extend(fetch_rows(config=config, split="train", offset=offset, length=min(100, total_rows - offset)))
    return pd.DataFrame(rows)


def chunked(values: Iterable[str], size: int = 20) -> Iterable[List[str]]:
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_matching_rows(config: str, column: str, ids: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for group in chunked(ids, size=20):
        where = " OR ".join('"{}"=\'{}\''.format(column, value.replace("'", "''")) for value in group)
        rows.extend(fetch_filter_rows(config=config, split="train", where=where))
    return pd.DataFrame(rows).drop_duplicates()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    hm_dir = output_dir / "hm_sample"
    amazon_dir = output_dir / "amazon_fashion"
    hm_dir.mkdir(parents=True, exist_ok=True)
    amazon_dir.mkdir(parents=True, exist_ok=True)

    transactions = paginated_rows("transactions", args.hm_transactions)
    customer_ids = transactions["customer_id"].drop_duplicates().head(80).tolist()
    article_ids = transactions["article_id"].astype(str).drop_duplicates().head(80).tolist()

    try:
        customers = fetch_matching_rows("customers", "customer_id", customer_ids)
    except Exception:
        customers = paginated_rows("customers", 100)
    try:
        articles = fetch_matching_rows("articles", "article_id", article_ids)
    except Exception:
        articles = paginated_rows("articles", 100)

    transactions.to_csv(hm_dir / "transactions_sample.csv", index=False, encoding="utf-8-sig")
    customers.to_csv(hm_dir / "customers_sample.csv", index=False, encoding="utf-8-sig")
    articles.to_csv(hm_dir / "articles_sample.csv", index=False, encoding="utf-8-sig")

    download_file(AMAZON_RATINGS_URL, amazon_dir / "ratings_Amazon_Fashion.csv")
    download_file(AMAZON_META_URL, amazon_dir / "meta_Amazon_Fashion.json.gz")

    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "sources": [
            {
                "name": "Amazon Fashion ratings",
                "url": AMAZON_RATINGS_URL,
                "destination": str((amazon_dir / "ratings_Amazon_Fashion.csv").resolve()),
            },
            {
                "name": "Amazon Fashion metadata",
                "url": AMAZON_META_URL,
                "destination": str((amazon_dir / "meta_Amazon_Fashion.json.gz").resolve()),
            },
            {
                "name": "H&M customers sample via dataset viewer",
                "url": "https://huggingface.co/datasets/{0}".format(HM_DATASET),
                "destination": str((hm_dir / "customers_sample.csv").resolve()),
            },
            {
                "name": "H&M articles sample via dataset viewer",
                "url": "https://huggingface.co/datasets/{0}".format(HM_DATASET),
                "destination": str((hm_dir / "articles_sample.csv").resolve()),
            },
            {
                "name": "H&M transactions sample via dataset viewer",
                "url": "https://huggingface.co/datasets/{0}".format(HM_DATASET),
                "destination": str((hm_dir / "transactions_sample.csv").resolve()),
            },
        ],
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Public data samples saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
