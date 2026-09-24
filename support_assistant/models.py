from pydantic import BaseModel, Field, ConfigDict


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
