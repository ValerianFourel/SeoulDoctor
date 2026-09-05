from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).parent))
from app import HashEncoder, Release, create_app
from build_fixture import build

def test_scoped_batch_and_limits(tmp_path):
    root=tmp_path/"release"; build(root)
    response=TestClient(create_app(Release(root,HashEncoder()))).post("/v1/retrieve",json={"facility_ids":["alpha"],"queries":[{"query_id":"q1","text":"friendly clinic"}],"limit_per_facility_per_channel":2})
    assert response.status_code == 200
    rows=response.json()["results"]
    assert {x["facility_id"] for x in rows} == {"alpha"}
    assert {x["channel"] for x in rows} == {"bge_m3_sparse","bge_m3_dense"}

def test_corrupt_artifact_fails_startup(tmp_path):
    root=tmp_path/"release"; build(root); (root/"reviews.jsonl").write_text("bad\n")
    with pytest.raises(Exception): Release(root,HashEncoder())
