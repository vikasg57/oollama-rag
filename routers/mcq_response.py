from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas import MCQResponseCreate, MCQResponseResponse
from crud import create_mcq_response, get_mcq_by_pdf, get_mcq_by_institution, get_all_mcqs
import uuid

router = APIRouter(prefix="/mcq-responses", tags=["MCQ Responses"])


@router.post("/", response_model=MCQResponseResponse)
def create_mcq(mcq: MCQResponseCreate, db: Session = Depends(get_db)):
    return create_mcq_response(db, mcq)


@router.get("/", response_model=list[MCQResponseResponse])
def list_mcqs(db: Session = Depends(get_db)):
    return get_all_mcqs(db)


@router.get("/by-pdf/{pdf_index_id}", response_model=list[MCQResponseResponse])
def list_mcqs_by_pdf(pdf_index_id: uuid.UUID, db: Session = Depends(get_db)):
    return get_mcq_by_pdf(db, pdf_index_id)


@router.get("/by-institution/{institution_id}", response_model=list[MCQResponseResponse])
def list_mcqs_by_institution(institution_id: uuid.UUID, db: Session = Depends(get_db)):
    return get_mcq_by_institution(db, institution_id)
