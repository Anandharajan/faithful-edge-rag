from pydantic import BaseModel


class ResearchProblemResponse(BaseModel):
    title: str
    problem: str

