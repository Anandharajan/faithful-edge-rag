from fastapi import APIRouter

from faithful_edge_rag.schemas.research import ResearchProblemResponse

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/problem", response_model=ResearchProblemResponse)
async def research_problem() -> ResearchProblemResponse:
    return ResearchProblemResponse(
        title="Open-Source Faithful Edge-Cloud RAG",
        problem=(
            "How can a fully open-source edge-cloud RAG system answer questions over "
            "distributed private knowledge while optimizing latency, compute cost, "
            "privacy exposure, and citation faithfulness under ambiguous or conflicting evidence?"
        ),
    )
