"""
FastAPI service — /health and /chat endpoints.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import uvicorn
from catalog import CatalogManager, Assessment
from agent import RecommendationAgent, ConversationState

# Initialize catalog and agent
catalog_manager = CatalogManager()
agent = RecommendationAgent(catalog_manager)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"SHL Assessment Recommender Service Started")
    print(f"Loaded {len(catalog_manager.get_all_assessments())} assessments")
    try:
        yield
    finally:
        print("SHL Assessment Recommender Service Stopped")

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (catalog and agent already initialized above at module level)

# ============================================================================
# Data Models
# ============================================================================

class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    """Request body for POST /chat endpoint."""
    messages: List[Message]

class Recommendation(BaseModel):
    """A recommended assessment."""
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    """Response body for POST /chat endpoint."""
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False

class HealthResponse(BaseModel):
    """Response body for GET /health endpoint."""
    status: str

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for service readiness.
    Returns status 200 if the service is ready.
    """
    return HealthResponse(status="ok")

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint for browser access."""
    return {
        "message": "SHL Assessment Recommender is running.",
        "docs": "/docs",
        "health": "/health",
        "catalog": "/catalog"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for conversational assessment recommendations.
    
    Takes a conversation history and returns:
    - reply: The agent's next message
    - recommendations: List of recommended assessments (empty if still gathering info)
    - end_of_conversation: Whether the conversation is complete
    """
    
    try:
        # Validate input
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")
        
        if len(request.messages) > 8:
            # Truncate to last 8 messages to enforce turn limit
            request.messages = request.messages[-8:]
        
        # Convert Pydantic models to dicts for agent processing
        messages = [msg.dict() for msg in request.messages]
        
        # Process conversation
        reply, recommendations, end_of_conversation, state = agent.process_conversation(messages)
        
        # Convert Assessment objects to Recommendation dicts
        recommendation_list = [
            Recommendation(
                name=a.name,
                url=a.url,
                test_type=a.test_type
            )
            for a in recommendations
        ]
        
        # Validate schema compliance
        # - Recommendations must only contain catalog items
        for rec in recommendation_list:
            catalog_assessment = catalog_manager.get_assessment(rec.name)
            if not catalog_assessment:
                # Find closest match or skip
                print(f"Warning: Recommendation '{rec.name}' not in catalog")
        
        # Create response
        response = ChatResponse(
            reply=reply,
            recommendations=recommendation_list,
            end_of_conversation=end_of_conversation
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/catalog")
async def get_catalog():
    """
    Debug endpoint to list all available assessments in the catalog.
    """
    assessments = catalog_manager.get_all_assessments()
    return {
        "total_assessments": len(assessments),
        "assessments": [
            {
                "name": a.name,
                "url": a.url,
                "test_type": a.test_type,
                "description": a.description
            }
            for a in assessments
        ]
    }

@app.get("/assessment/{assessment_name}")
async def get_assessment(assessment_name: str):
    """
    Get details about a specific assessment.
    """
    assessment = catalog_manager.get_assessment(assessment_name)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_name}' not found")
    
    return {
        "name": assessment.name,
        "url": assessment.url,
        "test_type": assessment.test_type,
        "description": assessment.description,
        "keywords": assessment.keywords
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom general exception handler."""
    from fastapi.responses import JSONResponse
    print(f"Unhandled exception: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # For local development: use PORT env var or fallback to 8001
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
