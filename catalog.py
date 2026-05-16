"""
Loads and searches the SHL product catalog.
Reads from shl_catalog.json (scraped from shl.com) on startup.
"""

import json
import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Map single-letter test type codes to human-readable labels
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Behavioral",
    "C": "Competency",
    "D": "Development",
    "E": "Exercise",
    "K": "Knowledge",
    "P": "Personality & Motivation",
    "S": "Simulation",
    "T": "Technical",
}


@dataclass
class Assessment:
    """Represents a single SHL assessment product."""
    name: str
    url: str
    test_type: str              # Primary type code (single letter)
    test_types: List[str] = field(default_factory=list)   # All type codes
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    remote_testing: bool = True
    adaptive: bool = False

    @property
    def test_type_label(self) -> str:
        """Human-readable label for the primary test type."""
        return TEST_TYPE_LABELS.get(self.test_type, self.test_type)

    @property
    def all_type_labels(self) -> List[str]:
        """Human-readable labels for all test types."""
        return [TEST_TYPE_LABELS.get(t, t) for t in self.test_types]


class CatalogManager:
    """Manages the SHL assessment catalog — loads real scraped data."""

    CATALOG_FILE = os.path.join(os.path.dirname(__file__), "shl_catalog.json")

    def __init__(self):
        self.assessments: List[Assessment] = []
        self._name_index: Dict[str, Assessment] = {}
        self._load_catalog()

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _load_catalog(self):
        """Load catalog from scraped JSON file, falling back to built-in data."""
        if os.path.exists(self.CATALOG_FILE):
            try:
                self._load_from_json(self.CATALOG_FILE)
                print(f"Catalog loaded: {len(self.assessments)} assessments from {self.CATALOG_FILE}")
                return
            except Exception as e:
                print(f"Could not load catalog JSON: {e}. Falling back to built-in data.")

        self._load_fallback_catalog()
        print(f"Using built-in fallback catalog ({len(self.assessments)} assessments)")

    def _load_from_json(self, path: str):
        """Load and parse the scraped catalog JSON."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.assessments = []
        for item in raw:
            name = item.get("name", "").strip()
            url = item.get("url", "").strip()
            if not name or not url:
                continue

            # Normalize URL to use shl.com domain
            if url.startswith("/"):
                url = "https://www.shl.com" + url
            elif not url.startswith("http"):
                url = "https://www.shl.com/" + url

            test_types = item.get("test_types", ["K"])
            if not test_types:
                test_types = ["K"]
            primary_type = test_types[0]

            description = item.get("description", "")
            if not description:
                description = self._generate_description(name, test_types)

            keywords = self._extract_keywords(name, description, test_types)

            assessment = Assessment(
                name=name,
                url=url,
                test_type=primary_type,
                test_types=test_types,
                description=description,
                keywords=keywords,
                remote_testing=item.get("remote_testing", True),
                adaptive=item.get("adaptive", False),
            )
            self.assessments.append(assessment)

        # Build name index for O(1) lookups
        self._build_index()

    def _build_index(self):
        self._name_index = {a.name.lower(): a for a in self.assessments}

    def _generate_description(self, name: str, test_types: List[str]) -> str:
        """Generate a meaningful description from name and type codes."""
        type_labels = [TEST_TYPE_LABELS.get(t, t) for t in test_types]
        type_str = " and ".join(type_labels[:2])
        # Clean up name for description
        clean_name = re.sub(r'\s*-\s*(Short Form|Solution|Assessment)\s*$', '', name, flags=re.I).strip()
        return f"{clean_name} — {type_str} assessment"

    def _extract_keywords(self, name: str, description: str, test_types: List[str]) -> List[str]:
        """Extract searchable keywords from assessment data."""
        combined = (name + " " + description).lower()
        keywords = set()

        # Add type labels as keywords
        for t in test_types:
            label = TEST_TYPE_LABELS.get(t, "").lower()
            if label:
                keywords.update(label.split())

        # Domain keyword list
        domain_kws = [
            # Technical
            "java", "python", "c++", "c#", "javascript", "typescript", "sql", "database",
            "cloud", "aws", "azure", "gcp", "devops", "linux", "windows", "network",
            "coding", "programming", "software", "developer", "engineer",
            # Cognitive
            "numerical", "verbal", "reasoning", "inductive", "deductive", "mechanical",
            "spatial", "checking", "attention", "detail", "critical", "analytical",
            # Behavioral
            "personality", "motivation", "behavior", "teamwork", "communication",
            "leadership", "management", "stakeholder", "sales", "customer", "service",
            # Roles
            "manager", "analyst", "administrator", "agent", "clerk", "supervisor",
            "executive", "professional", "graduate", "apprentice", "associate",
            # Industries
            "banking", "finance", "retail", "healthcare", "call center", "contact center",
            "accounting", "bookkeeping", "operations", "administrative",
        ]
        for kw in domain_kws:
            if kw in combined:
                keywords.add(kw)

        # Always add significant words from the name itself
        for word in re.findall(r'\b[a-zA-Z]{3,}\b', name.lower()):
            if word not in {"the", "and", "for", "with", "new", "form", "test", "short"}:
                keywords.add(word)

        return sorted(keywords)

    # -------------------------------------------------------------------------
    # Search & retrieval
    # -------------------------------------------------------------------------

    def search_assessments(self, query: str, filters: Dict = None) -> List["Assessment"]:
        """
        Full-text search over assessments.
        Returns results sorted by relevance (most matches first).
        """
        results_with_score = []
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]

        for assessment in self.assessments:
            score = self._search_score(assessment, query_lower, query_words)
            if score > 0:
                results_with_score.append((assessment, score))

        results_with_score.sort(key=lambda x: x[1], reverse=True)
        results = [a for a, _ in results_with_score]

        # Apply filters
        if filters:
            if "test_type" in filters:
                results = [a for a in results if filters["test_type"] in a.test_types]
            if "exclude_types" in filters:
                results = [a for a in results if not any(t in a.test_types for t in filters["exclude_types"])]

        return results

    def _search_score(self, assessment: Assessment, query_lower: str, query_words: List[str]) -> float:
        """Score an assessment's relevance to a query."""
        score = 0.0
        name_lower = assessment.name.lower()
        desc_lower = assessment.description.lower()
        kw_text = " ".join(assessment.keywords)

        # Exact name match (highest weight)
        if query_lower in name_lower:
            score += 10

        # Word-by-word matches
        for word in query_words:
            if word in name_lower:
                score += 4
            if word in desc_lower:
                score += 2
            if word in kw_text:
                score += 3

        return score

    def get_assessment(self, name: str) -> Optional[Assessment]:
        """Get assessment by exact name (case-insensitive)."""
        # Exact match
        result = self._name_index.get(name.lower())
        if result:
            return result

        # Partial match fallback
        name_lower = name.lower()
        for a in self.assessments:
            if name_lower in a.name.lower() or a.name.lower() in name_lower:
                return a
        return None

    def get_all_assessments(self) -> List[Assessment]:
        return self.assessments

    def get_by_type(self, type_code: str) -> List[Assessment]:
        """Get all assessments of a specific type."""
        return [a for a in self.assessments if type_code in a.test_types]

    def get_assessment_details(self, assessment: Assessment) -> Dict:
        return {
            "name": assessment.name,
            "url": assessment.url,
            "test_type": assessment.test_type,
            "test_type_label": assessment.test_type_label,
            "all_types": assessment.test_types,
            "all_type_labels": assessment.all_type_labels,
            "description": assessment.description,
            "keywords": assessment.keywords,
            "remote_testing": assessment.remote_testing,
            "adaptive": assessment.adaptive,
        }

    # -------------------------------------------------------------------------
    # Built-in fallback (used if JSON file missing)
    # -------------------------------------------------------------------------

    def _load_fallback_catalog(self):
        """Minimal built-in catalog as emergency fallback."""
        fallback = [
            Assessment("Verify - Numerical Reasoning", "https://www.shl.com/products/product-catalog/view/verify-numerical-reasoning/", "A", ["A"],
                       "Numerical reasoning and data interpretation ability", ["numerical", "reasoning", "math", "quantitative"]),
            Assessment("Verify - Verbal Reasoning", "https://www.shl.com/products/product-catalog/view/verify-verbal-reasoning/", "A", ["A"],
                       "Verbal reasoning and comprehension", ["verbal", "reasoning", "language", "reading"]),
            Assessment("Verify - Inductive Reasoning", "https://www.shl.com/products/product-catalog/view/verify-inductive-reasoning/", "A", ["A"],
                       "Inductive and abstract reasoning", ["inductive", "reasoning", "abstract", "pattern"]),
            Assessment("OPQ32r", "https://www.shl.com/products/product-catalog/view/opq32r/", "P", ["P"],
                       "Comprehensive personality questionnaire measuring 32 scales", ["personality", "behavior", "traits", "motivation"]),
            Assessment("MQ - Motivation Questionnaire", "https://www.shl.com/products/product-catalog/view/mq-motivation-questionnaire/", "P", ["P"],
                       "Motivation and engagement assessment", ["motivation", "engagement", "values"]),
            Assessment("Java 8 (New)", "https://www.shl.com/products/product-catalog/view/java-8-new/", "K", ["K"],
                       "Java programming knowledge", ["java", "programming", "technical", "developer"]),
            Assessment("Python (New)", "https://www.shl.com/products/product-catalog/view/python-new/", "K", ["K"],
                       "Python programming knowledge", ["python", "programming", "technical", "developer"]),
            Assessment("SQL (New)", "https://www.shl.com/products/product-catalog/view/sql-new/", "K", ["K"],
                       "SQL and database knowledge", ["sql", "database", "data"]),
            Assessment("Verify - Checking", "https://www.shl.com/products/product-catalog/view/verify-checking/", "A", ["A"],
                       "Attention to detail and error-checking", ["checking", "attention", "detail"]),
            Assessment("General Ability - Short Form", "https://www.shl.com/products/product-catalog/view/general-ability-short-form/", "A", ["A", "K"],
                       "General cognitive ability", ["general", "ability", "cognitive", "reasoning"]),
        ]
        self.assessments = fallback
        self._build_index()
