# schemas.py
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID


# -------------------- PDFIndex --------------------
class PDFIndexBase(BaseModel):
    filename: str
    index_name: str
    institution_id: UUID  # Use UUID type


class PDFIndexCreate(PDFIndexBase):
    pass


class PDFIndexResponse(PDFIndexBase):
    id: UUID
    uploaded_at: datetime

    class Config:
        orm_mode = True


# -------------------- Institution --------------------
class InstitutionBase(BaseModel):
    name: str


class InstitutionCreate(InstitutionBase):
    api_request_limit: int = 100


class InstitutionResponse(InstitutionBase):
    id: UUID
    api_request_count: int
    api_request_limit: int

    class Config:
        orm_mode = True


# -------------------- MCQResponse --------------------
class MCQResponseBase(BaseModel):
    institution_id: UUID
    pdf_index_id: UUID
    mcq_data: str


class MCQResponseCreate(MCQResponseBase):
    pass


class MCQResponseResponse(MCQResponseBase):
    id: UUID
    uploaded_at: datetime

    class Config:
        orm_mode = True
