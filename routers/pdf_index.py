import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas import PDFIndexCreate, PDFIndexResponse
from crud import create_pdf_index, get_all_pdf_indices, delete_pdf_index

router = APIRouter(prefix="/pdfs", tags=["PDF Indices"])


@router.post("/", response_model=PDFIndexResponse)
def create_pdf(pdf: PDFIndexCreate, db: Session = Depends(get_db)):
    return create_pdf_index(db, pdf)


@router.get("/", response_model=list[PDFIndexResponse])
def list_pdfs(db: Session = Depends(get_db)):
    return get_all_pdf_indices(db)


@router.delete("/{pdf_id}", response_model=bool)
def delete_pdf(db: Session = Depends(get_db), pdf_id: uuid.UUID = None):
    return delete_pdf_index(db, pdf_id)
