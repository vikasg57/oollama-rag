import uuid
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base
from sqlalchemy import Text  # Import Text if not already


class PDFIndex(Base):
    __tablename__ = "pdf_indices"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String, unique=True, index=True)
    index_name: Mapped[str] = mapped_column(String)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    institution_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("institutions.id"))
    institution: Mapped["Institution"] = relationship("Institution", back_populates="pdf_indices")

    mcq_responses: Mapped[list["MCQResponse"]] = relationship("MCQResponse", back_populates="pdf_index")


class Institution(Base):
    __tablename__ = "institutions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    api_request_count: Mapped[int] = mapped_column(Integer, default=0)
    api_request_limit: Mapped[int] = mapped_column(Integer, default=100)

    # 🔥 FIX: You had institution_id pointing to institutions.id (which is itself). Removed.
    mcq_responses: Mapped[list["MCQResponse"]] = relationship("MCQResponse", back_populates="institution")
    pdf_indices: Mapped[list["PDFIndex"]] = relationship("PDFIndex", back_populates="institution")


class MCQResponse(Base):
    __tablename__ = "mcq_responses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    mcq_data: Mapped[str] = mapped_column(Text)  # Store raw Markdown as string

    institution_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("institutions.id"))
    pdf_index_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pdf_indices.id"))

    institution: Mapped["Institution"] = relationship("Institution", back_populates="mcq_responses")
    pdf_index: Mapped["PDFIndex"] = relationship("PDFIndex", back_populates="mcq_responses")
