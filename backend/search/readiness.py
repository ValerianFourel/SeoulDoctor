"""Exercise retrieval and reranking, independent of web-server liveness."""

from time import perf_counter

from search.semantic_retriever import SemanticCellQuery


def probe_retrieval(scoped_index, semantic_source, reranker):
    started = perf_counter()
    report = {"ready": False, "gpu_execution_verified": False}
    if semantic_source is None or reranker is None:
        return {**report, "reason": "required_service_not_configured"}
    try:
        samples = scoped_index.search_evidence(
            "진료 설명 consultation", limit=1, source_types=("verbatim_review",),
        )
        if not samples:
            return {**report, "reason": "no_probe_evidence"}
        facility_id = samples[0].facility_id
        retrieval_started = perf_counter()
        outcome = semantic_source.retrieve(
            review_source_sha256=scoped_index.review_source_sha256,
            facility_ids=(facility_id,),
            queries=(SemanticCellQuery("readiness", "explanation", "support", "ko", "진료 설명"),),
            limit_per_facility=2,
        )
        report["retrieval_ms"] = (perf_counter() - retrieval_started) * 1000
        if not outcome.used:
            return {**report, "reason": "semantic_" + outcome.status}
        if {item.channel for item in outcome.references} != {"bge_m3_dense", "bge_m3_sparse"}:
            return {**report, "reason": "missing_retrieval_channel"}
        ids = tuple(dict.fromkeys(item.evidence_id for item in outcome.references))
        resolved = scoped_index.resolve_evidence_ids(ids)
        if ({item.evidence_id for item in resolved} != set(ids)
                or any(item.facility_id != facility_id for item in resolved)):
            return {**report, "reason": "invalid_evidence_ownership"}
        rerank_started = perf_counter()
        reranked = reranker.rerank("진료 설명 consultation", resolved)
        report["reranking_ms"] = (perf_counter() - rerank_started) * 1000
        if not reranked.used or {key for key, _ in reranked.scores} != set(ids):
            return {**report, "reason": "reranker_" + reranked.reason}
        return {**report, "ready": True, "reason": "operations_verified",
                "elapsed_ms": (perf_counter() - started) * 1000}
    except Exception as exc:
        # Exception messages may contain endpoint addresses or credentials.
        return {**report, "reason": "probe_failed", "error_type": type(exc).__name__}
