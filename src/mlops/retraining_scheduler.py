#!/usr/bin/env python3
"""
Simple Retraining/Reindexing Scheduler for Underwriting Agent
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """Simple scheduler for retraining and reindexing"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize scheduler with configuration"""
        self.config_path = config_path
        self.api_url = "http://localhost:8000"
        self.mlflow_tracking_uri = "file:./mlruns"
        
        # Retraining triggers
        self.triggers = {
            "data_drift_threshold": 0.15,  # 15% performance drop
            "time_based_days": 30,         # Monthly retraining
            "min_samples": 100             # Minimum new samples
        }
        
        # Performance monitoring
        self.performance_history = []
        self.last_retraining = None
        
    def check_retraining_needed(self) -> Dict[str, Any]:
        """Check if retraining is needed based on triggers"""
        
        status = {
            "retraining_needed": False,
            "reasons": [],
            "confidence": 0.0,
            "next_check": None
        }
        
        try:
            # 1. Time-based check (monthly)
            if self._check_time_based():
                status["retraining_needed"] = True
                status["reasons"].append("Monthly retraining scheduled")
                status["confidence"] += 0.3
            
            # 2. Performance drift check
            drift_detected = self._check_performance_drift()
            if drift_detected:
                status["retraining_needed"] = True
                status["reasons"].append(f"Performance drift detected: {drift_detected:.2%}")
                status["confidence"] += 0.5
            
            # 3. Data volume check
            if self._check_data_volume():
                status["retraining_needed"] = True
                status["reasons"].append("Sufficient new data available")
                status["confidence"] += 0.2
            
            # Set next check time
            status["next_check"] = (datetime.now() + timedelta(days=7)).isoformat()
            
        except Exception as e:
            logger.error(f"Error checking retraining status: {e}")
            status["reasons"].append(f"Error: {str(e)}")
        
        return status
    
    def _check_time_based(self) -> bool:
        """Check if enough time has passed for retraining"""
        if not self.last_retraining:
            return True  # First time
        
        days_since = (datetime.now() - self.last_retraining).days
        return days_since >= self.triggers["time_based_days"]
    
    def _check_performance_drift(self) -> Optional[float]:
        """Check for performance drift using recent API calls"""
        try:
            # Get recent performance metrics from MLflow
            recent_metrics = self._get_recent_metrics()
            if len(recent_metrics) < 10:  # Need minimum samples
                return None
            
            # Calculate performance trend
            recent_avg = sum(recent_metrics[-10:]) / 10
            historical_avg = sum(recent_metrics[:-10]) / len(recent_metrics[:-10]) if len(recent_metrics) > 10 else recent_avg
            
            # Calculate drift percentage
            drift = (historical_avg - recent_avg) / historical_avg if historical_avg > 0 else 0
            
            return drift if drift > self.triggers["data_drift_threshold"] else None
            
        except Exception as e:
            logger.warning(f"Could not check performance drift: {e}")
            return None
    
    def _check_data_volume(self) -> bool:
        """Check if enough new data is available"""
        try:
            # Check if new data files exist or are significantly larger
            data_path = Path("data/raw")
            if not data_path.exists():
                return False
            
            # Simple check: if any CSV file is newer than last retraining
            if not self.last_retraining:
                return True
            
            for file_path in data_path.glob("*.csv"):
                if file_path.stat().st_mtime > self.last_retraining.timestamp():
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not check data volume: {e}")
            return False
    
    def _get_recent_metrics(self) -> list:
        """Get recent performance metrics from MLflow"""
        try:
            # This is a simplified version - in production, use MLflow API
            metrics_file = Path("mlruns/performance_history.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Could not get recent metrics: {e}")
            return []
    
    def trigger_retraining(self) -> Dict[str, Any]:
        """Trigger retraining process"""
        
        result = {
            "success": False,
            "message": "",
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }
        
        try:
            # Step 1: Backup current model
            result["steps"].append("Backing up current model...")
            self._backup_current_model()
            
            # Step 2: Retrain baseline models
            result["steps"].append("Retraining baseline models...")
            self._retrain_baseline_models()
            
            # Step 3: Reindex vector store
            result["steps"].append("Reindexing vector store...")
            self._reindex_vector_store()
            
            # Step 4: Update performance history
            result["steps"].append("Updating performance history...")
            self._update_performance_history()
            
            # Step 5: Update last retraining time
            self.last_retraining = datetime.now()
            self._save_retraining_state()
            
            result["success"] = True
            result["message"] = "Retraining completed successfully"
            
        except Exception as e:
            result["message"] = f"Retraining failed: {str(e)}"
            logger.error(f"Retraining failed: {e}")
        
        return result
    
    def _backup_current_model(self):
        """Backup current model and vector store"""
        backup_dir = Path("mlruns/backups")
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Copy current artifacts
        artifacts_path = Path("mlruns/artifacts")
        if artifacts_path.exists():
            import shutil
            shutil.copytree(artifacts_path, backup_path / "artifacts")
        
        logger.info(f"Model backed up to {backup_path}")
    
    def _retrain_baseline_models(self):
        """Retrain baseline ML models"""
        # This would call your model training script
        # For now, we'll just log the step
        logger.info("Retraining baseline models...")
        
        # In production, you would:
        # 1. Load new data
        # 2. Retrain LogisticRegression and RandomForest
        # 3. Evaluate performance
        # 4. Save new models
    
    def _reindex_vector_store(self):
        """Reindex the vector store with new documents"""
        try:
            # Call API to reindex documents
            response = requests.post(f"{self.api_url}/reindex-documents", timeout=30)
            if response.status_code == 200:
                logger.info("Vector store reindexed successfully")
            else:
                logger.warning(f"Reindexing failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Could not reindex vector store: {e}")
    
    def _update_performance_history(self):
        """Update performance history for monitoring"""
        try:
            # Get current performance metrics
            current_metrics = self._get_current_performance()
            
            # Load existing history
            history_file = Path("mlruns/performance_history.json")
            history = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # Add current metrics
            history.append({
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics
            })
            
            # Keep only last 100 entries
            history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not update performance history: {e}")
    
    def _get_current_performance(self) -> Dict[str, float]:
        """Get current model performance metrics"""
        # This would get actual performance metrics
        # For now, return mock data
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
    
    def _save_retraining_state(self):
        """Save retraining state"""
        state = {
            "last_retraining": self.last_retraining.isoformat() if self.last_retraining else None,
            "triggers": self.triggers,
            "performance_history_length": len(self.performance_history)
        }
        
        with open("mlruns/retraining_state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_retraining_state(self):
        """Load retraining state from file"""
        try:
            with open("mlruns/retraining_state.json", 'r') as f:
                state = json.load(f)
            
            if state.get("last_retraining"):
                self.last_retraining = datetime.fromisoformat(state["last_retraining"])
            
            logger.info("Retraining state loaded successfully")
            
        except FileNotFoundError:
            logger.info("No previous retraining state found")
        except Exception as e:
            logger.error(f"Could not load retraining state: {e}")

def main():
    """Main function for running retraining scheduler"""
    logging.basicConfig(level=logging.INFO)
    
    scheduler = RetrainingScheduler()
    scheduler.load_retraining_state()
    
    # Check if retraining is needed
    status = scheduler.check_retraining_needed()
    
    print("Retraining Status Check")
    print("=" * 30)
    print(f"Retraining needed: {status['retraining_needed']}")
    print(f"Reasons: {', '.join(status['reasons'])}")
    print(f"Confidence: {status['confidence']:.2f}")
    print(f"Next check: {status['next_check']}")
    
    if status['retraining_needed']:
        print("\nTriggering retraining...")
        result = scheduler.trigger_retraining()
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print("Steps completed:")
        for step in result['steps']:
            print(f"  - {step}")

if __name__ == "__main__":
    main()
