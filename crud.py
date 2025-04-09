from sqlalchemy.orm import Session
from models import PDFIndex, Institution, MCQResponse
from schemas import PDFIndexCreate, InstitutionCreate, MCQResponseCreate
import uuid
from fastapi import HTTPException


# -------------------- PDFIndex --------------------
def create_pdf_index(db: Session, pdf_data: PDFIndexCreate):
    institution = db.query(Institution).filter(Institution.id == pdf_data.institution_id).first()
    if not institution:
        raise ValueError("Institution not found")

    db_pdf = PDFIndex(
        filename=pdf_data.filename,
        index_name=pdf_data.index_name,
        institution_id=pdf_data.institution_id
    )
    db.add(db_pdf)
    db.commit()
    db.refresh(db_pdf)
    return db_pdf


def get_all_pdf_indices(db: Session):
    return db.query(PDFIndex).all()


def get_all_pdf_indices_for_institution(db: Session, institution_id: uuid.UUID):
    return db.query(PDFIndex).filter(institution_id == institution_id)


def get_pdf_index_by_id(db: Session, pdf_id: uuid.UUID):
    return db.query(PDFIndex).filter(PDFIndex.id == pdf_id).first()


def delete_pdf_index(db: Session, pdf_id: uuid.UUID):
    db_pdf = get_pdf_index_by_id(db, pdf_id)
    if db_pdf:
        db.delete(db_pdf)
        db.commit()
        return True
    return False


# -------------------- Institution --------------------
def create_institution(db: Session, institution_data: InstitutionCreate):
    db_inst = Institution(**institution_data.dict())
    db.add(db_inst)
    db.commit()
    db.refresh(db_inst)
    return db_inst


def get_all_institutions(db: Session):
    return db.query(Institution).all()


def get_institution_by_id(db: Session, inst_id: uuid.UUID):
    return db.query(Institution).filter(Institution.id == inst_id).first()


def update_institution_limit(db: Session, inst_id: uuid.UUID, new_limit: int):
    db_inst = get_institution_by_id(db, inst_id)
    if db_inst:
        db_inst.api_request_limit = new_limit
        db.commit()
        db.refresh(db_inst)
        return db_inst
    return None


def increment_api_request_count(db: Session, institution_id: uuid.UUID, increment_by: int = 1):
    institution = db.query(Institution).filter(Institution.id == institution_id).first()
    if not institution:
        raise HTTPException(status_code=404, detail="Institution not found")

    institution.api_request_count += increment_by
    db.commit()
    db.refresh(institution)
    return institution


# -------------------- MCQResponse --------------------
def create_mcq_response(db: Session, mcq_data: MCQResponseCreate):
    # Optional validation to ensure institution/pdf exist
    institution = db.query(Institution).filter(Institution.id == mcq_data.institution_id).first()
    if not institution:
        raise ValueError("Institution not found")
    pdf_index = db.query(PDFIndex).filter(PDFIndex.id == mcq_data.pdf_index_id).first()
    if not pdf_index:
        raise ValueError("PDF Index not found")

    pdf = db.query(PDFIndex).filter(PDFIndex.id == mcq_data.pdf_index_id).first()
    if not pdf:
        raise ValueError("PDF Index not found")

    db_mcq = MCQResponse(
        institution_id=mcq_data.institution_id,
        pdf_index_id=mcq_data.pdf_index_id,
        mcq_data=mcq_data.mcq_data,
    )
    db.add(db_mcq)
    db.commit()
    db.refresh(db_mcq)
    return db_mcq


def get_mcq_by_pdf(db: Session, pdf_index_id: uuid.UUID):
    return db.query(MCQResponse).filter(MCQResponse.pdf_index_id == pdf_index_id).all()


def get_mcq_by_institution(db: Session, institution_id: uuid.UUID):
    return db.query(MCQResponse).filter(MCQResponse.institution_id == institution_id).all()


def get_all_mcqs(db: Session):
    return db.query(MCQResponse).all()
