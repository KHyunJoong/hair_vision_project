import json
from unicodedata import category

from fastapi import APIRouter

from app.models.problems_response import Problem, ProblemsResponse
from app.models.create_problems_request import CreateProblemsRequest
# from app.models.info import Info
from app.services.question_maker import category_question_generation
import logging


router = APIRouter()
logger = logging.getLogger("uvicorn")


@router.post("/hair")
async def check_hair(request: CreateProblemsRequest) -> ProblemsResponse:

    note = request.note
    logger.info("check_hair start")
    logger.info(f"note : {note}")

    problems,low_categories = category_question_generation(note)
    categoriescount = len(low_categories)
    logger.info(f"problems : {problems}")
    logger.info(f"categoriescount : {categoriescount}")

    categories = set([p['category'] for p in problems])
    logger.info(f"categories : {categories}")

    response = ProblemsResponse(
        count=len(problems),
        categories=list(categories),
        problems=[Problem(**p) for p in problems]
    )
    logger.info(f"response : {response}")

    return response
