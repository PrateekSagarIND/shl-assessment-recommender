"""
Conversation logic for SHL assessment recommendations.
Handles clarification, recommendation, refinement, comparison and refusals.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from catalog import Assessment, CatalogManager

# Try Gemini first (new SDK), fall back to rule-based
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ConversationState(Enum):
    CLARIFYING = "clarifying"
    RECOMMENDING = "recommending"
    COMPARING = "comparing"
    REFINING = "refining"
    DONE = "done"


@dataclass
class ConversationContext:
    role_type: Optional[str] = None
    seniority_level: Optional[str] = None
    skills_focus: List[str] = None
    personality_required: bool = False
    technical_required: bool = False
    behavioral_required: bool = False
    job_description: Optional[str] = None
    specific_keywords: List[str] = None

    def __post_init__(self):
        if self.skills_focus is None:
            self.skills_focus = []
        if self.specific_keywords is None:
            self.specific_keywords = []
        if self.job_description is None:
            self.job_description = ""


class RecommendationAgent:
    """Handles conversation routing and generates responses."""

    def __init__(self, catalog_manager: CatalogManager):
        self.catalog = catalog_manager
        self.max_turns = 8
        self.llm_client = None
        self.llm_type = None
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client — tries OpenAI first, then Gemini, then rule-based fallback."""
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

        # Try OpenAI first — 2.5M tokens/day on gpt-4o-mini
        if OPENAI_AVAILABLE and openai_key:
            try:
                self.llm_client = OpenAI(api_key=openai_key)
                self.llm_type = "openai"
                self.openai_model = "gpt-4o-mini"
                print(f"LLM: using OpenAI {self.openai_model}")
                return
            except Exception as e:
                print(f"OpenAI init failed: {e}")

        # Fallback to Gemini
        if GEMINI_AVAILABLE and gemini_key:
            try:
                self.llm_client = google_genai.Client(api_key=gemini_key)
                self.llm_type = "gemini"
                self.gemini_model = "models/gemini-flash-latest"
                print(f"LLM: using Gemini {self.gemini_model}")
                return
            except Exception as e:
                print(f"Gemini init failed: {e}")

        print("No LLM API key found, using rule-based fallback")
        self.llm_type = "rule_based"

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def process_conversation(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, List[Assessment], bool, ConversationState]:
        """
        Takes full conversation history, returns:
        (reply, recommendations, end_of_conversation, state)
        """
        if not messages:
            return "Hello! I'm the SHL Assessment Recommender. What role are you hiring for?", [], False, ConversationState.CLARIFYING

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # Hard scope check first
        if self._is_out_of_scope(last_user_msg):
            return (
                "I can only help with SHL assessment recommendations. "
                "Please tell me about the role you're hiring for and I'll find the right assessments.",
                [],
                False,
                ConversationState.CLARIFYING,
            )

        # Route to LLM or rule-based
        if self.llm_type in ("gemini", "openai"):
            return self._process_with_llm(messages, last_user_msg)
        else:
            return self._process_rule_based(messages, last_user_msg)

    # -------------------------------------------------------------------------
    # LLM-powered processing
    # -------------------------------------------------------------------------

    def _process_with_llm(
        self, messages: List[Dict[str, str]], last_user_msg: str
    ) -> Tuple[str, List[Assessment], bool, ConversationState]:
        """Use LLM to decide intent and generate response, grounded in catalog data."""

        # Step 1: Classify intent
        intent = self._classify_intent(messages, last_user_msg)

        # Step 2: Extract context from full history
        context = self._extract_context(messages)

        if intent == "compare":
            reply = self._llm_compare(messages, last_user_msg)
            return reply, [], False, ConversationState.COMPARING

        elif intent in ("recommend", "refine"):
            # Search catalog with extracted context
            candidates = self._get_recommendations(context)
            if not candidates or len(context.specific_keywords) + (1 if context.role_type else 0) < 1:
                # Not enough info yet
                reply = self._llm_clarify(messages, context)
                return reply, [], False, ConversationState.CLARIFYING

            reply = self._llm_recommendation_reply(context, candidates, is_refinement=(intent == "refine"))
            state = ConversationState.REFINING if intent == "refine" else ConversationState.RECOMMENDING
            return reply, candidates, True, state

        else:  # clarify
            reply = self._llm_clarify(messages, context)
            return reply, [], False, ConversationState.CLARIFYING

    def _classify_intent(self, messages: List[Dict[str, str]], last_user_msg: str) -> str:
        """Classify the user's intent: clarify | recommend | refine | compare."""
        history_str = self._format_history(messages[:-1])
        has_prior_recommendations = any(
            "recommend" in m.get("content", "").lower() or "here are" in m.get("content", "").lower()
            for m in messages
            if m["role"] == "assistant"
        )

        prompt = f"""You are classifying user intent in a conversation about SHL assessment recommendations.

Conversation so far:
{history_str}

Latest user message: "{last_user_msg}"

Classify the intent as exactly one of:
- "compare" — user wants to compare/contrast specific assessments
- "recommend" — user has given enough info and wants recommendations  
- "refine" — user is updating/changing previous recommendation request
- "clarify" — agent needs more info before recommending

Rules:
- "recommend" only if we know the role AND at least one qualifier (seniority, skills, tech stack, etc.)
- "clarify" if the query is too vague ("I need an assessment", "help me")
- Prior recommendations exist: {has_prior_recommendations}

Reply with ONLY the single word: compare, recommend, refine, or clarify"""

        try:
            response = self._call_llm(prompt)
            intent = response.strip().lower().split()[0]
            if intent in ("compare", "recommend", "refine", "clarify"):
                return intent
        except Exception as e:
            print(f"Intent classification error: {e}")

        # Fallback: rule-based classification
        msg_lower = last_user_msg.lower()
        if any(w in msg_lower for w in ["compare", "difference", "vs", "versus", "which is better"]):
            return "compare"
        if has_prior_recommendations and any(w in msg_lower for w in ["also", "add", "include", "change", "actually", "instead"]):
            return "refine"
        context = self._extract_context(messages)
        if context.role_type and (context.seniority_level or context.specific_keywords):
            return "recommend"
        return "clarify"

    def _llm_clarify(self, messages: List[Dict[str, str]], context: ConversationContext) -> str:
        """Generate a natural clarifying question using the LLM."""
        history_str = self._format_history(messages)
        known_info = []
        if context.role_type:
            known_info.append(f"role: {context.role_type}")
        if context.seniority_level:
            known_info.append(f"seniority: {context.seniority_level}")
        if context.skills_focus:
            known_info.append(f"skills focus: {', '.join(context.skills_focus)}")

        prompt = f"""You are an SHL assessment recommendation agent helping a hiring manager find the right assessments.

Conversation:
{history_str}

Known context so far: {', '.join(known_info) if known_info else 'nothing yet'}

Ask ONE focused clarifying question to gather the most important missing info needed to recommend assessments.
Priority order for what to ask:
1. Role/job title (if unknown)
2. Seniority level (if unknown)  
3. Key skills to assess (technical, personality, behavioral)

Be concise, professional, and friendly. Ask only ONE question."""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"LLM clarify error: {e}")
            return self._generate_clarification(context)

    def _llm_recommendation_reply(
        self, context: ConversationContext, recommendations: List[Assessment], is_refinement: bool = False
    ) -> str:
        """Generate a natural recommendation reply grounded in catalog data."""
        assessment_list = "\n".join(
            f"- {a.name} ({a.test_type_label}): {a.description}"
            for a in recommendations[:10]
        )
        action = "updated" if is_refinement else "selected"
        prompt = f"""You are an SHL assessment recommendation agent.

You have {action} these assessments for a {context.role_type or 'role'}{(' at ' + context.seniority_level + ' level') if context.seniority_level else ''}:

{assessment_list}

Write a short, professional reply of 2-3 sentences introducing these recommendations.
Do not use bullet points. Do not add any meta-commentary. Output only the reply text itself.
Mention why they suit the role, then ask if the user wants refinements or comparisons."""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"LLM recommendation reply error: {e}")
            count = len(recommendations)
            role = context.role_type or "this role"
            return f"Here are {count} assessments tailored for a {role}. They cover the key competencies you mentioned. Would you like to refine this list or compare any assessments?"

    def _llm_compare(self, messages: List[Dict[str, str]], last_user_msg: str) -> str:
        """Compare two or more assessments using catalog data + LLM."""
        # Extract assessment names mentioned
        all_names = [a.name for a in self.catalog.get_all_assessments()]

        # Find which assessments are mentioned
        mentioned = []
        msg_lower = last_user_msg.lower()
        for name in all_names:
            if name.lower() in msg_lower or any(part.lower() in msg_lower for part in name.split() if len(part) > 3):
                mentioned.append(name)

        if len(mentioned) < 2:
            # Try to extract from previous messages too
            full_text = " ".join(m["content"] for m in messages).lower()
            for name in all_names:
                if name.lower() in full_text and name not in mentioned:
                    mentioned.append(name)

        mentioned = mentioned[:3]  # Compare up to 3

        if len(mentioned) < 2:
            return ("Could you specify which assessments you'd like to compare? "
                   "For example: 'What's the difference between OPQ32r and WAVE?'")

        # Build comparison data from catalog
        comparison_data = []
        for name in mentioned:
            a = self.catalog.get_assessment(name)
            if a:
                comparison_data.append(
                    f"**{a.name}** (Type: {a.test_type_label}): {a.description}. "
                    f"Keywords: {', '.join(a.keywords[:6])}"
                )

        prompt = f"""You are an SHL assessment expert. Compare these assessments based ONLY on the data provided:

{chr(10).join(comparison_data)}

User question: "{last_user_msg}"

Write a concise, grounded comparison (3-5 sentences). Explain what each measures and when to choose each one.
Do NOT make up information not present in the data above."""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"LLM compare error: {e}")
            if len(mentioned) >= 2:
                a1 = self.catalog.get_assessment(mentioned[0])
                a2 = self.catalog.get_assessment(mentioned[1])
                if a1 and a2:
                    return self._create_comparison_text(a1, a2)
            return "I can compare specific assessments — please name the two you'd like to compare."

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM with retry on rate-limit (429)."""
        import time
        if self.llm_type == "gemini":
            model = getattr(self, "gemini_model", "gemini-1.5-flash")
            for attempt in range(3):
                try:
                    response = self.llm_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            max_output_tokens=400, temperature=0.3
                        ),
                    )
                    return response.text.strip()
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = (attempt + 1) * 10
                        print(f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError("Gemini rate limit exceeded after retries")
        elif self.llm_type == "openai":
            model = getattr(self, "openai_model", "gpt-4o-mini")
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        raise RuntimeError("No LLM client available")

    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for LLM prompts."""
        lines = []
        for m in messages:
            role = "User" if m["role"] == "user" else "Agent"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines) if lines else "(empty)"

    # -------------------------------------------------------------------------
    # Context extraction
    # -------------------------------------------------------------------------

    def _extract_context(self, messages: List[Dict[str, str]]) -> ConversationContext:
        context = ConversationContext()
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        combined_text = " ".join(user_messages).lower()

        # Role detection
        role_map = {
            "java developer": "Java Developer",
            "python developer": "Python Developer",
            "c++ developer": "C++ Developer",
            "c# developer": "C# Developer",
            "javascript developer": "JavaScript Developer",
            "frontend developer": "Frontend Developer",
            "backend developer": "Backend Developer",
            "full stack developer": "Full Stack Developer",
            "data scientist": "Data Scientist",
            "data analyst": "Data Analyst",
            "data engineer": "Data Engineer",
            "product manager": "Product Manager",
            "project manager": "Project Manager",
            "software engineer": "Software Engineer",
            "devops engineer": "DevOps Engineer",
            "cloud engineer": "Cloud Engineer",
            "machine learning engineer": "Machine Learning Engineer",
            "manager": "Manager",
            "team lead": "Team Lead",
            "sales": "Sales",
            "customer service": "Customer Service",
            "accountant": "Accountant",
            "finance": "Finance",
            "hr": "HR",
            "recruiter": "Recruiter",
            "java": "Java Developer",
            "python": "Python Developer",
            "developer": "Developer",
            "engineer": "Engineer",
            "analyst": "Analyst",
        }
        for kw, role in role_map.items():
            if kw in combined_text:
                context.role_type = role
                break

        # Regex fallback for "hiring a X" pattern
        if not context.role_type:
            m = re.search(r'hiring\s+(?:a\s+|an\s+)?([\w\s]+?)(?:\s+who|\s+with|\s+for|$)', combined_text)
            if m:
                candidate = m.group(1).strip()
                if candidate and candidate not in {"someone", "person", "people", "candidate"}:
                    context.role_type = candidate.title()

        # Seniority
        if any(w in combined_text for w in ["junior", "entry level", "entry-level", "graduate", "fresher", "new grad"]):
            context.seniority_level = "junior"
        elif any(w in combined_text for w in ["senior", "principal", "staff", "lead", "expert"]):
            context.seniority_level = "senior"
        elif any(w in combined_text for w in ["mid", "intermediate", "mid-level", "3 years", "4 years", "5 years", "6 years"]):
            context.seniority_level = "mid-level"

        # Assessment type needs
        if any(w in combined_text for w in ["personality", "traits", "character", "values", "motivation"]):
            context.personality_required = True
            context.skills_focus.append("personality")
        if any(w in combined_text for w in ["technical", "coding", "programming", "technology", "software"]):
            context.technical_required = True
            context.skills_focus.append("technical")
        if any(w in combined_text for w in ["behavior", "teamwork", "communication", "stakeholder", "leadership", "interpersonal"]):
            context.behavioral_required = True
            context.skills_focus.append("behavioral")
        if any(w in combined_text for w in ["numerical", "math", "quantitative", "numbers"]):
            context.skills_focus.append("numerical")
        if any(w in combined_text for w in ["verbal", "language", "reading", "writing"]):
            context.skills_focus.append("verbal")
        if any(w in combined_text for w in ["reasoning", "logic", "critical thinking", "analytical"]):
            context.skills_focus.append("reasoning")

        context.job_description = " ".join(user_messages)
        context.specific_keywords = self._extract_keywords(combined_text)
        return context

    def _extract_keywords(self, text: str) -> List[str]:
        kw_list = [
            "stakeholder", "leadership", "communication", "teamwork", "java", "python", "c++",
            "database", "cloud", "aws", "azure", "gcp", "analytical", "problem-solving",
            "technical", "management", "reasoning", "attention", "detail", "data", "sql",
            "numerical", "verbal", "checking", "personality", "behavioral", "motivation",
            "sales", "customer", "finance", "javascript", "typescript", "devops", "agile",
            "scrum", "machine learning", "ai", "deep learning", "nlp",
        ]
        return [kw for kw in kw_list if kw in text]

    # -------------------------------------------------------------------------
    # Recommendation engine
    # -------------------------------------------------------------------------

    def _get_recommendations(self, context: ConversationContext) -> List[Assessment]:
        query_parts = []
        if context.role_type:
            query_parts.append(context.role_type)
        query_parts.extend(context.specific_keywords)
        query = " ".join(query_parts) or "assessment"

        candidates = self.catalog.search_assessments(query)

        # Type filtering
        if context.personality_required and not context.technical_required:
            personality_candidates = [a for a in candidates if a.test_type == "P"]
            if personality_candidates:
                candidates = personality_candidates
        elif context.technical_required and not context.personality_required:
            tech_candidates = [a for a in candidates if a.test_type in ["T", "K", "S"]]
            if tech_candidates:
                candidates = tech_candidates

        # Score and rank
        scored = [(a, self._score_assessment(a, context)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [a for a, score in scored if score > 0]

        # Fallback: return top scored even if score is 0
        if not results:
            results = [a for a, _ in scored[:10]]

        return results[:10]

    def _score_assessment(self, assessment: Assessment, context: ConversationContext) -> float:
        score = 0.0
        role_lower = (context.role_type or "").lower()
        assess_text = (assessment.name + " " + assessment.description + " " + " ".join(assessment.keywords)).lower()

        # Role name match
        for word in role_lower.split():
            if len(word) > 2 and word in assess_text:
                score += 6

        # Seniority bonus (general assessments work for all levels)
        if context.seniority_level:
            score += 1

        # Keyword matches
        for kw in context.specific_keywords:
            if kw in assess_text:
                score += 3

        # Type preference match
        if context.personality_required and assessment.test_type == "P":
            score += 5
        if context.technical_required and assessment.test_type in ["T", "K", "S"]:
            score += 5
        if context.behavioral_required and assessment.test_type in ["B", "P"]:
            score += 3

        # Developer role → prioritize knowledge/technical tests
        if context.role_type and "developer" in role_lower and assessment.test_type in ["T", "K"]:
            score += 4

        # Stakeholder/communication → personality + behavioral
        if "stakeholder" in context.specific_keywords and assessment.test_type in ["P", "B"]:
            score += 4

        return score

    # -------------------------------------------------------------------------
    # Rule-based fallback (no LLM)
    # -------------------------------------------------------------------------

    def _process_rule_based(
        self, messages: List[Dict[str, str]], last_user_msg: str
    ) -> Tuple[str, List[Assessment], bool, ConversationState]:
        context = self._extract_context(messages)
        state = self._determine_state(context, messages)

        if state == ConversationState.COMPARING:
            reply = self._generate_comparison(messages)
            return reply, [], False, state
        elif state == ConversationState.RECOMMENDING:
            recs = self._get_recommendations(context)
            reply = self._generate_recommendation_reply(context, recs)
            return reply, recs, True, state
        elif state == ConversationState.REFINING:
            recs = self._get_recommendations(context)
            reply = self._generate_refinement_reply(context, recs)
            return reply, recs, True, state
        else:
            reply = self._generate_clarification(context)
            return reply, [], False, ConversationState.CLARIFYING

    def _determine_state(self, context: ConversationContext, messages: List[Dict[str, str]]) -> ConversationState:
        last_user = next((m["content"].lower() for m in reversed(messages) if m["role"] == "user"), "")
        if any(w in last_user for w in ["compare", "difference", "vs", "versus", "which is better"]):
            return ConversationState.COMPARING

        has_role = bool(context.role_type)
        has_qualifier = bool(context.seniority_level) or len(context.specific_keywords) > 0
        has_prior = any(
            m["role"] == "assistant" and ("here are" in m["content"].lower() or "assessment" in m["content"].lower())
            for m in messages
        )

        if has_role and has_qualifier:
            return ConversationState.REFINING if has_prior else ConversationState.RECOMMENDING
        return ConversationState.CLARIFYING

    def _is_out_of_scope(self, msg: str) -> bool:
        msg_lower = msg.lower()
        out_of_scope = [
            "legal", "lawsuit", "gdpr", "discrimination", "salary negotiation",
            "hiring law", "employment law", "benefit", "stock option",
            "how to hire", "recruiting strategy", "ignore previous",
            "forget your instructions", "pretend you are", "jailbreak",
            "act as", "new persona", "ignore all previous"
        ]
        return any(kw in msg_lower for kw in out_of_scope)

    def _generate_clarification(self, context: ConversationContext) -> str:
        if not context.role_type:
            return "What role are you hiring for? (e.g., Java Developer, Product Manager, Data Analyst)"
        if not context.seniority_level:
            return f"What seniority level for the {context.role_type}? (junior, mid-level, or senior)"
        if not context.skills_focus:
            return (f"What competencies matter most for this {context.role_type}? "
                    "For example: technical skills, personality, reasoning ability, or behavioral traits?")
        return f"Any other specific requirements for the {context.role_type} role?"

    def _generate_recommendation_reply(self, context: ConversationContext, recs: List[Assessment]) -> str:
        if not recs:
            return f"I couldn't find specific assessments for {context.role_type}. Could you tell me more about the required skills?"
        role = context.role_type or "this role"
        level = f" at {context.seniority_level} level" if context.seniority_level else ""
        return (f"Here are {len(recs)} assessments that fit a {role}{level}. "
                "They cover the key competencies you mentioned. Would you like to refine this list?")

    def _generate_refinement_reply(self, context: ConversationContext, recs: List[Assessment]) -> str:
        role = context.role_type or "this role"
        return (f"Updated! Here are {len(recs)} assessments better matching your requirements for the {role}. "
                "Let me know if you'd like further adjustments or comparisons.")

    def _generate_comparison(self, messages: List[Dict[str, str]]) -> str:
        last_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        all_names = [a.name for a in self.catalog.get_all_assessments()]
        found = [n for n in all_names if n.lower() in last_msg.lower()]
        if len(found) >= 2:
            a1 = self.catalog.get_assessment(found[0])
            a2 = self.catalog.get_assessment(found[1])
            if a1 and a2:
                return self._create_comparison_text(a1, a2)
        return "Which two assessments would you like to compare? Please mention their names."

    def _create_comparison_text(self, a1: Assessment, a2: Assessment) -> str:
        return (
            f"{a1.name} vs {a2.name}:\n\n"
            f"**{a1.name}** (Type: {a1.test_type_label}): {a1.description}. Keywords: {', '.join(a1.keywords[:5])}\n\n"
            f"**{a2.name}** (Type: {a2.test_type_label}): {a2.description}. Keywords: {', '.join(a2.keywords[:5])}\n\n"
            f"Choose {a1.name} when you need to assess {a1.description.lower()}. "
            f"Choose {a2.name} when you need to assess {a2.description.lower()}."
        )
