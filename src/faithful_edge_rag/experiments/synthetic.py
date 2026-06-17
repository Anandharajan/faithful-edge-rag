from faithful_edge_rag.experiments.models import DocumentChunk, QueryCase

TOPICS: tuple[tuple[str, str, str, str], ...] = (
    (
        "battery_overheat",
        "Battery overheating mitigation threshold",
        "If pack temperature exceeds 45 C, throttle charging and trigger fan mode.",
        "If pack temperature exceeds 50 C, continue charging under observation.",
    ),
    (
        "firmware_rollback",
        "Firmware rollback policy",
        "Rollback firmware when checksum drift is detected twice within 10 minutes.",
        "Rollback firmware only after five checksum drift events.",
    ),
    (
        "pump_vibration",
        "Pump vibration shutdown rule",
        "Stop the pump when vibration RMS is above 7.5 mm/s for 30 seconds.",
        "Stop the pump when vibration RMS is above 10.0 mm/s for 60 seconds.",
    ),
    (
        "gateway_cache",
        "Gateway cache retention policy",
        "Keep edge retrieval cache for 72 hours during intermittent connectivity.",
        "Keep edge retrieval cache for 24 hours during intermittent connectivity.",
    ),
    (
        "sensor_calibration",
        "Sensor calibration schedule",
        "Calibrate pressure sensors every 14 days after firmware v3.2.",
        "Calibrate pressure sensors every 30 days after firmware v3.2.",
    ),
    (
        "privacy_sync",
        "Privacy-preserving sync policy",
        "Sync summaries and provenance only for local-only inspection records.",
        "Sync full raw inspection records for all maintenance events.",
    ),
    (
        "valve_latency",
        "Valve latency alarm threshold",
        "Raise a valve latency alarm when actuation delay exceeds 180 ms.",
        "Raise a valve latency alarm when actuation delay exceeds 300 ms.",
    ),
    (
        "compressor_restart",
        "Compressor restart policy",
        "Wait 12 minutes before compressor restart after thermal shutdown.",
        "Wait 5 minutes before compressor restart after thermal shutdown.",
    ),
)


def build_synthetic_corpus() -> tuple[list[DocumentChunk], list[QueryCase]]:
    chunks: list[DocumentChunk] = []
    queries: list[QueryCase] = []
    edge_nodes = ("edge-a", "edge-b")

    for idx, (topic, title, current_claim, stale_claim) in enumerate(TOPICS, start=1):
        edge_node_id = edge_nodes[idx % len(edge_nodes)]
        private = idx % 3 == 0

        current_id = f"{topic}-current"
        stale_id = f"{topic}-stale"
        summary_id = f"{topic}-summary"
        noise_id = f"{topic}-noise"

        chunks.extend(
            [
                DocumentChunk(
                    chunk_id=current_id,
                    topic=topic,
                    text=(
                        f"{title}. Current field directive version 3. "
                        f"{current_claim} This directive applies at {edge_node_id}."
                    ),
                    claim=current_claim,
                    edge_node_id=edge_node_id,
                    version=3,
                    authority=3,
                    is_private=private,
                    is_summary=False,
                    token_count=42,
                ),
                DocumentChunk(
                    chunk_id=stale_id,
                    topic=topic,
                    text=(
                        f"{title}. Archived directive version 1. "
                        f"{stale_claim} This older document is retained for audit."
                    ),
                    claim=stale_claim,
                    edge_node_id=edge_node_id,
                    version=1,
                    authority=1,
                    is_private=False,
                    is_summary=False,
                    token_count=40,
                ),
                DocumentChunk(
                    chunk_id=summary_id,
                    topic=topic,
                    text=(
                        f"{title}. Edge summary says the active version is 3 and "
                        f"the approved claim is: {current_claim}"
                    ),
                    claim=current_claim,
                    edge_node_id=edge_node_id,
                    version=3,
                    authority=2,
                    is_private=False,
                    is_summary=True,
                    token_count=25,
                ),
                DocumentChunk(
                    chunk_id=noise_id,
                    topic=topic,
                    text=(
                        f"{title}. Maintenance note discusses inventory, shift handoff, "
                        "and weather delays but does not define the active rule."
                    ),
                    claim="No operational rule stated.",
                    edge_node_id=edge_node_id,
                    version=2,
                    authority=0,
                    is_private=False,
                    is_summary=False,
                    token_count=35,
                ),
            ]
        )
        queries.append(
            QueryCase(
                query_id=f"q-{topic}",
                topic=topic,
                question=f"What is the current approved rule for {title.lower()}?",
                edge_node_id=edge_node_id,
                expected_claim=current_claim,
                relevant_chunk_ids=(current_id, summary_id),
                conflict_expected=True,
                requires_private_evidence=private,
            )
        )

    return chunks, queries
