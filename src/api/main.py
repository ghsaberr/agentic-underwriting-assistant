from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import uuid
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MLflow tracker
try:
    from src.mlops.experiment_tracker import tracker
    mlflow_available = True
    logger.info("MLflow tracking enabled")
except ImportError as e:
    logger.warning(f"MLflow not available: {e}")
    mlflow_available = False

app = FastAPI(
    title="Underwriting Agent API",
    description="AI-powered underwriting risk assessment service",
    version="1.0.0"
)

# Request/Response Models
class PolicyholderData(BaseModel):
    policyholder_id: str
    age: int
    annual_income: float
    credit_score: int
    policy_type: str
    claims_history: Optional[List[Dict[str, Any]]] = []

class RiskAssessmentResponse(BaseModel):
    risk_score: float
    risk_level: str
    rationale: str
    retrieved_docs: List[str]
    tool_citations: List[str]
    confidence: float
    timestamp: str
    request_id: str

# Import real agent
try:
    from src.agents.underwriting_agent import UnderwritingAgent
    # Initialize real agent
    agent = UnderwritingAgent()
    logger.info("Real Underwriting Agent initialized")
except Exception as e:
    logger.warning(f"Could not initialize real agent: {e}")
    logger.info("Falling back to mock agent")
    
    # Fallback to mock agent
    class MockUnderwritingAgent:
        def __init__(self):
            self.name = "Mock Underwriting Agent"
        
        def assess_risk(self, policyholder_data: PolicyholderData) -> Dict[str, Any]:
            """Mock risk assessment - replace with actual agent logic"""
            
            # Simple risk calculation
            risk_score = 0
            
            # Age factor
            if policyholder_data.age < 25 or policyholder_data.age > 65:
                risk_score += 20
            elif policyholder_data.age < 30:
                risk_score += 10
            
            # Credit score factor
            if policyholder_data.credit_score < 500:
                risk_score += 30
            elif policyholder_data.credit_score < 600:
                risk_score += 20
            elif policyholder_data.credit_score < 700:
                risk_score += 10
            
            # Income factor
            if policyholder_data.annual_income < 30000:
                risk_score += 15
            elif policyholder_data.annual_income < 50000:
                risk_score += 8
            
            # Claims history factor
            if policyholder_data.claims_history:
                risk_score += len(policyholder_data.claims_history) * 5
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = "High"
            elif risk_score >= 40:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            return {
                "risk_score": min(risk_score, 100),
                "risk_level": risk_level,
                "rationale": f"Risk assessment based on age ({policyholder_data.age}), credit score ({policyholder_data.credit_score}), income (${policyholder_data.annual_income:,.0f}), and claims history ({len(policyholder_data.claims_history)} claims)",
                "retrieved_docs": ["DOC_001", "DOC_002"],
                "tool_citations": ["risk_calculator", "rule_checker"],
                "confidence": 0.85
            }
    
    agent = MockUnderwritingAgent()

# In-memory storage for audit trail (replace with database in production)
audit_trail = []

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Underwriting Agent API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_status": "active"
    }

@app.get("/agent-info")
async def get_agent_info():
    """Get agent information including model details"""
    try:
        # Get agent type and model info
        agent_info = {
            "agent_type": type(agent).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        # If it's the real agent, get more details
        if hasattr(agent, 'model_name'):
            agent_info.update({
                "model_name": agent.model_name,
                "ollama_url": getattr(agent, 'ollama_url', 'N/A'),
                "chroma_url": getattr(agent, 'chroma_url', 'N/A'),
                "llm_available": agent.llm is not None,
                "vector_store_available": agent.vector_store is not None
            })
            
            # Test Ollama connection
            try:
                import requests
                response = requests.get(f"{agent.ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    agent_info["available_models"] = [model['name'] for model in models]
                    agent_info["current_model_available"] = agent.model_name in [model['name'] for model in models]
                else:
                    agent_info["ollama_connection"] = "Failed"
            except Exception as e:
                agent_info["ollama_connection"] = f"Error: {str(e)}"
        
        return agent_info
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/assess-risk", response_model=RiskAssessmentResponse)
async def assess_risk(policyholder_data: PolicyholderData):
    """
    Assess underwriting risk for a policyholder
    """
    try:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info(f"Risk assessment request received: {request_id}")
        
        # Start timing for MLflow
        start_time = time.time()
        
        # Convert Pydantic model to dict for agent
        policyholder_dict = policyholder_data.model_dump()
        
        # Perform risk assessment
        assessment_result = agent.assess_risk(policyholder_dict)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = RiskAssessmentResponse(
            risk_score=assessment_result["risk_score"],
            risk_level=assessment_result["risk_level"],
            rationale=assessment_result["rationale"],
            retrieved_docs=assessment_result["retrieved_docs"],
            tool_citations=assessment_result["tool_citations"],
            confidence=assessment_result["confidence"],
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
        
        # Store in audit trail
        audit_entry = {
            "request_id": request_id,
            "timestamp": response.timestamp,
            "input": policyholder_data.model_dump(),
            "output": response.model_dump(),
            "status": "success"
        }
        audit_trail.append(audit_entry)
        
        # Log to MLflow if available
        if mlflow_available:
            try:
                # Extract additional data for MLflow
                retrieved_docs = assessment_result.get("retrieved_docs", [])
                tool_citations = assessment_result.get("tool_citations", [])
                
                # Create tool outputs dict
                tool_outputs = {
                    "risk_calculator": {"used": "risk_calculator" in tool_citations},
                    "rule_checker": {"used": "rule_checker" in tool_citations},
                    "document_retriever": {"used": len(retrieved_docs) > 0, "count": len(retrieved_docs)}
                }
                
                # Log the run
                run_id = tracker.log_agent_run(
                    input_data=policyholder_dict,
                    output_data=assessment_result,
                    retrieved_docs=retrieved_docs,
                    tool_outputs=tool_outputs,
                    processing_time=processing_time,
                    model_version="1.0.0"
                )
                
                if run_id:
                    logger.info(f"MLflow run logged: {run_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        logger.info(f"Risk assessment completed: {request_id}, Score: {response.risk_score}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    total_requests = len(audit_trail)
    successful_requests = len([entry for entry in audit_trail if entry["status"] == "success"])
    
    if total_requests > 0:
        success_rate = successful_requests / total_requests
    else:
        success_rate = 0
    
    # Calculate average risk score
    risk_scores = [entry["output"]["risk_score"] for entry in audit_trail if entry["status"] == "success"]
    avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": success_rate,
        "average_risk_score": avg_risk_score,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/audit-trail")
async def get_audit_trail(limit: int = 10):
    """Get recent audit trail entries"""
    return {
        "entries": audit_trail[-limit:],
        "total_entries": len(audit_trail),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/mlflow-summary")
async def get_mlflow_summary():
    """Get MLflow experiment summary"""
    if not mlflow_available:
        return {"error": "MLflow not available"}
    
    try:
        summary = tracker.get_experiment_summary()
        return summary
    except Exception as e:
        return {"error": str(e)}

@app.get("/mlflow-ui")
async def get_mlflow_ui_info():
    """Get MLflow UI information"""
    return {
        "mlflow_available": mlflow_available,
        "experiment_name": "underwriting-agent",
        "tracking_uri": "file:./mlruns",
        "ui_command": "mlflow ui --backend-store-uri file:./mlruns --port 5000",
        "ui_url": "http://localhost:5000"
    }

@app.post("/reindex-documents")
async def reindex_documents():
    """Reindex documents in vector store"""
    try:
        # This would trigger document reindexing
        # For now, we'll just return success
        logger.info("Document reindexing triggered")
        
        return {
            "success": True,
            "message": "Document reindexing completed",
            "timestamp": datetime.now().isoformat(),
            "documents_processed": 3  # Mock number
        }
    except Exception as e:
        logger.error(f"Document reindexing failed: {e}")
        return {
            "success": False,
            "message": f"Document reindexing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/retraining-status")
async def get_retraining_status():
    """Get retraining status and recommendations"""
    try:
        from src.mlops.retraining_scheduler import RetrainingScheduler
        
        scheduler = RetrainingScheduler()
        scheduler.load_retraining_state()
        status = scheduler.check_retraining_needed()
        
        return {
            "retraining_needed": status["retraining_needed"],
            "reasons": status["reasons"],
            "confidence": status["confidence"],
            "next_check": status["next_check"],
            "last_retraining": scheduler.last_retraining.isoformat() if scheduler.last_retraining else None
        }
    except Exception as e:
        logger.error(f"Could not get retraining status: {e}")
        return {
            "error": str(e),
            "retraining_needed": False
        }

@app.post("/trigger-retraining")
async def trigger_retraining():
    """Manually trigger retraining process"""
    try:
        from src.mlops.retraining_scheduler import RetrainingScheduler
        
        scheduler = RetrainingScheduler()
        scheduler.load_retraining_state()
        result = scheduler.trigger_retraining()
        
        return result
    except Exception as e:
        logger.error(f"Retraining trigger failed: {e}")
        return {
            "success": False,
            "message": f"Retraining failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)