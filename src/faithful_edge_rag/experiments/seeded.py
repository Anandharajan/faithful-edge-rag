from random import Random

from faithful_edge_rag.experiments.models import DocumentChunk, QueryCase

NOUNS = (
    "battery",
    "pump",
    "gateway",
    "sensor",
    "valve",
    "compressor",
    "inverter",
    "router",
    "coolant",
    "actuator",
)
PROPERTIES = (
    "temperature threshold",
    "rollback rule",
    "vibration limit",
    "cache retention",
    "calibration interval",
    "sync policy",
    "latency alarm",
    "restart delay",
    "pressure limit",
    "inspection interval",
)
ACTIONS = (
    "throttle charging",
    "rollback firmware",
    "stop the pump",
    "keep local cache",
    "calibrate sensors",
    "sync summaries only",
    "raise an alarm",
    "wait before restart",
    "open a maintenance ticket",
    "isolate the device",
)
UNITS = ("C", "events", "mm/s", "hours", "days", "records", "ms", "minutes", "bar", "cycles")


def build_seeded_corpus(
    *, seed: int, topics: int = 120, edge_nodes: int = 4
) -> tuple[list[DocumentChunk], list[QueryCase]]:
    rng = Random(seed)
    chunks: list[DocumentChunk] = []
    queries: list[QueryCase] = []

    for index in range(topics):
        noun = rng.choice(NOUNS)
        prop = rng.choice(PROPERTIES)
        action = rng.choice(ACTIONS)
        unit = rng.choice(UNITS)
        topic = f"{noun}_{prop.replace(' ', '_')}_{index:03d}"
        edge_node_id = f"edge-{index % edge_nodes}"
        private = rng.random() < 0.35
        conflict = rng.random() < 0.72
        current_value = rng.randint(10, 220)
        stale_delta = rng.choice((-20, -10, 10, 20, 40))
        stale_value = max(1, current_value + stale_delta)
        current_claim = f"{action} when {prop} reaches {current_value} {unit}."
        stale_claim = f"{action} when {prop} reaches {stale_value} {unit}."
        title = f"{noun.title()} {prop}"

        current_id = f"{topic}-current"
        summary_id = f"{topic}-summary"
        chunks.append(
            DocumentChunk(
                chunk_id=current_id,
                topic=topic,
                text=(
                    f"{title}. Current field directive version 3 for {edge_node_id}. "
                    f"The approved rule is to {current_claim}"
                ),
                claim=current_claim,
                edge_node_id=edge_node_id,
                version=3,
                authority=3,
                is_private=private,
                is_summary=False,
                token_count=rng.randint(34, 52),
            )
        )
        chunks.append(
            DocumentChunk(
                chunk_id=summary_id,
                topic=topic,
                text=(
                    f"{title}. Edge summary for {edge_node_id}: active version 3 says "
                    f"{current_claim}"
                ),
                claim=current_claim,
                edge_node_id=edge_node_id,
                version=3,
                authority=2,
                is_private=False,
                is_summary=True,
                token_count=rng.randint(18, 31),
            )
        )

        relevant = [current_id, summary_id]
        if conflict:
            stale_id = f"{topic}-stale"
            chunks.append(
                DocumentChunk(
                    chunk_id=stale_id,
                    topic=topic,
                    text=(
                        f"{title}. Archived directive version 1 for audit. "
                        f"The retired rule was to {stale_claim}"
                    ),
                    claim=stale_claim,
                    edge_node_id=edge_node_id,
                    version=1,
                    authority=1,
                    is_private=False,
                    is_summary=False,
                    token_count=rng.randint(32, 50),
                )
            )

        for noise_index in range(rng.randint(1, 3)):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{topic}-noise-{noise_index}",
                    topic=topic,
                    text=(
                        f"{title}. Maintenance note {noise_index} describes staffing, "
                        "weather, inventory, and shift handoff without setting the rule."
                    ),
                    claim="No operational rule stated.",
                    edge_node_id=edge_node_id,
                    version=2,
                    authority=0,
                    is_private=False,
                    is_summary=False,
                    token_count=rng.randint(25, 44),
                )
            )

        queries.append(
            QueryCase(
                query_id=f"q-{topic}",
                topic=topic,
                question=(
                    f"What is the current approved rule for {title.lower()} "
                    f"at {edge_node_id}?"
                ),
                edge_node_id=edge_node_id,
                expected_claim=current_claim,
                relevant_chunk_ids=tuple(relevant),
                conflict_expected=conflict,
                requires_private_evidence=private,
            )
        )

    return chunks, queries
