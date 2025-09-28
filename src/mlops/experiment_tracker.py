#!/usr/bin/env python3
"""
MLflow Experiment Tracking for Underwriting Agent
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """MLflow experiment tracker for underwriting agent"""
    
    def __init__(self, experiment_name: str = "underwriting-agent"):
        """Initialize MLflow tracker"""
        self.experiment_name = experiment_name
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow experiment"""
        try:
            # Prefer env var if provided; default to shared absolute path inside containers
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            self.experiment_id = None
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run"""
        if self.experiment_id is None:
            logger.warning("MLflow not properly initialized")
            return None
        
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=tags or {}
            )
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None
    
    def log_agent_run(self, 
                     input_data: Dict[str, Any],
                     output_data: Dict[str, Any],
                     retrieved_docs: List[Dict],
                     tool_outputs: Dict[str, Any],
                     processing_time: float,
                     model_version: str = "1.0.0"):
        """Log a complete agent run"""
        
        run = self.start_run(
            run_name=f"agent_run_{datetime.now().strftime('%H%M%S')}",
            tags={
                "type": "agent_run",
                "model_version": model_version,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        if run is None:
            return None
        
        try:
            with run:
                # Log parameters
                mlflow.log_params({
                    "policyholder_id": input_data.get("policyholder_id", "unknown"),
                    "age": input_data.get("age", 0),
                    "credit_score": input_data.get("credit_score", 0),
                    "annual_income": input_data.get("annual_income", 0),
                    "policy_type": input_data.get("policy_type", "unknown")
                })
                
                # Log metrics
                mlflow.log_metrics({
                    "risk_score": output_data.get("risk_score", 0),
                    "confidence": output_data.get("confidence", 0),
                    "processing_time_seconds": processing_time,
                    "retrieved_docs_count": len(retrieved_docs),
                    "tools_used_count": len(tool_outputs)
                })
                
                # Log artifacts (detailed data)
                artifacts_dir = f"./mlruns/artifacts/{run.info.run_id}"
                os.makedirs(artifacts_dir, exist_ok=True)
                
                # Save input data
                with open(f"{artifacts_dir}/input_data.json", "w") as f:
                    json.dump(input_data, f, indent=2, default=str)
                
                # Save output data
                with open(f"{artifacts_dir}/output_data.json", "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                # Save retrieved documents
                with open(f"{artifacts_dir}/retrieved_docs.json", "w") as f:
                    json.dump(retrieved_docs, f, indent=2, default=str)
                
                # Save tool outputs
                with open(f"{artifacts_dir}/tool_outputs.json", "w") as f:
                    json.dump(tool_outputs, f, indent=2, default=str)
                
                # Log artifacts
                mlflow.log_artifacts(artifacts_dir, "run_details")
                
                logger.info(f"Logged agent run: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error logging agent run: {e}")
            return None
    
    def log_model_performance(self,
                            model_name: str,
                            model_type: str,
                            metrics: Dict[str, float],
                            parameters: Dict[str, Any],
                            model_path: str = None):
        """Log model performance metrics"""
        
        run = self.start_run(
            run_name=f"model_eval_{model_name}_{datetime.now().strftime('%H%M%S')}",
            tags={
                "type": "model_evaluation",
                "model_name": model_name,
                "model_type": model_type
            }
        )
        
        if run is None:
            return None
        
        try:
            with run:
                # Log parameters
                mlflow.log_params(parameters)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model if path provided
                if model_path and os.path.exists(model_path):
                    if model_type == "sklearn":
                        mlflow.sklearn.log_model(model_path, "model")
                    else:
                        mlflow.log_artifact(model_path, "model")
                
                logger.info(f"Logged model performance: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error logging model performance: {e}")
            return None
    
    def log_embedding_experiment(self,
                                embedding_model: str,
                                document_count: int,
                                embedding_dim: int,
                                retrieval_accuracy: float,
                                processing_time: float):
        """Log embedding experiment results"""
        
        run = self.start_run(
            run_name=f"embedding_exp_{datetime.now().strftime('%H%M%S')}",
            tags={
                "type": "embedding_experiment",
                "embedding_model": embedding_model
            }
        )
        
        if run is None:
            return None
        
        try:
            with run:
                mlflow.log_params({
                    "embedding_model": embedding_model,
                    "document_count": document_count,
                    "embedding_dimension": embedding_dim
                })
                
                mlflow.log_metrics({
                    "retrieval_accuracy": retrieval_accuracy,
                    "processing_time_seconds": processing_time,
                    "documents_per_second": document_count / processing_time if processing_time > 0 else 0
                })
                
                logger.info(f"Logged embedding experiment: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error logging embedding experiment: {e}")
            return None
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {"error": "Experiment not found"}
            
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            summary = {
                "experiment_name": self.experiment_name,
                "total_runs": len(runs),
                "run_types": runs.get("tags.type", pd.Series()).value_counts().to_dict() if not runs.empty else {},
                "latest_runs": runs.head(5).to_dict("records") if not runs.empty else []
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting experiment summary: {e}")
            return {"error": str(e)}

# Global tracker instance
tracker = ExperimentTracker()
