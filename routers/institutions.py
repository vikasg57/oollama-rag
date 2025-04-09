from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas import InstitutionCreate, InstitutionResponse
from crud import create_institution, get_all_institutions, get_institution_by_id, update_institution_limit
import uuid

router = APIRouter(prefix="/institutions", tags=["Institutions"])


@router.post("/", response_model=InstitutionResponse)
def create_new_institution(institution: InstitutionCreate, db: Session = Depends(get_db)):
    return create_institution(db, institution)


@router.get("/", response_model=list[InstitutionResponse])
def list_all_institutions(db: Session = Depends(get_db)):
    return get_all_institutions(db)


@router.get("/{inst_id}", response_model=InstitutionResponse)
def get_institution(inst_id: uuid.UUID, db: Session = Depends(get_db)):
    inst = get_institution_by_id(db, inst_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Institution not found")
    return inst


@router.put("/{inst_id}/limit", response_model=InstitutionResponse)
def update_limit(inst_id: uuid.UUID, new_limit: int, db: Session = Depends(get_db)):
    updated = update_institution_limit(db, inst_id, new_limit)
    if not updated:
        raise HTTPException(status_code=404, detail="Institution not found")
    return updated
