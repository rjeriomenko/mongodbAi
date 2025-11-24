from __future__ import annotations

from pydantic import BaseModel


class Company(BaseModel):
    """Company data model"""
    company_title: str
    company_website_url: str | None = None
    employee_emails: list[str] = []
    culture: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {
            "company_title": self.company_title,
            "company_website_url": self.company_website_url,
            "employee_emails": self.employee_emails,
            "culture": self.culture,
        }

    @classmethod
    def from_mongo_doc(cls, doc: dict) -> "Company":
        """Create Company from MongoDB document."""
        return cls(
            company_title=doc.get("company_title", ""),
            company_website_url=doc.get("company_website_url"),
            employee_emails=doc.get("employee_emails", []),
            culture=doc.get("culture"),
        )
