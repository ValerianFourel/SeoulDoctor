"""Build a deterministic tiny release for CI and local smoke tests."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    reviews=[{"evidence_id":"e-alpha-1","facility_id":"alpha","text":"friendly clinic"},{"evidence_id":"e-alpha-2","facility_id":"alpha","text":"친절한 병원"},{"evidence_id":"e-bravo-1","facility_id":"bravo","text":"ordinary clinic"}]
    (output/"reviews.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in reviews),encoding="utf-8")
    (output/"facility_ranges.json").write_text(json.dumps({"alpha":[0,2],"bravo":[2,3]},sort_keys=True))
    sparse=[{"friendly":1.0,"clinic":1.0},{"친절한":1.0,"병원":1.0},{"ordinary":1.0,"clinic":1.0}]
    (output/"sparse.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in sparse),encoding="utf-8")
    np.save(output/"dense.npy",np.asarray([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]],dtype="<f4"))
    def rec(name):
        p=output/name; return {"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest={"schema":"seouldoc.bge-m3-review-manifest/v1","release_id":"fixture-v1","model_id":"BAAI/bge-m3","model_revision":"fixture","review_source_sha256":rec("reviews.jsonl")["sha256"],"review_count":3,"dimension":8,"artifacts":{n:rec(n) for n in ("reviews.jsonl","facility_ranges.json","dense.npy","sparse.jsonl")}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))

if __name__ == "__main__": build(Path(sys.argv[1] if len(sys.argv)>1 else "release"))
