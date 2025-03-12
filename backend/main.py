import json
from unicodedata import category

from fastapi import APIRouter

from app.models.problems_response import Problem, ProblemsResponse
from app.models.create_problems_request import CreateProblemsRequest
# from app.models.info import Info
from app.services.question_maker import category_question_generation
import logging

from fastapi import FastAPI

from app.api.endpoints import problems
from app.core.config import settings


app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(problems.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
