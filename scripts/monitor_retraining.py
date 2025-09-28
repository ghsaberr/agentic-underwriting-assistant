#!/usr/bin/env python3
"""
Simple monitoring script for retraining/reindexing
Run this as a cron job: 0 9 * * 1 (every Monday at 9 AM)
"""

import sys
import os
import logging
import requests
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlops.retraining_scheduler import RetrainingScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retraining_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False

def send_alert(message: str, level: str = "INFO"):
    """Send alert (in production, integrate with Slack/Email)"""
    logger.info(f"ALERT [{level}]: {message}")
    
    # In production, you would:
    # - Send Slack notification
    # - Send email
    # - Create JIRA ticket
    # - etc.

def main():
    """Main monitoring function"""
    logger.info("Starting retraining monitoring check...")
    
    # Check API health
    if not check_api_health():
        send_alert("API is not responding", "ERROR")
        return
    
    # Initialize scheduler
    scheduler = RetrainingScheduler()
    scheduler.load_retraining_state()
    
    # Check retraining status
    status = scheduler.check_retraining_needed()
    
    logger.info(f"Retraining needed: {status['retraining_needed']}")
    logger.info(f"Reasons: {', '.join(status['reasons'])}")
    logger.info(f"Confidence: {status['confidence']:.2f}")
    
    # If retraining is needed, trigger it
    if status['retraining_needed']:
        send_alert(f"Retraining needed: {', '.join(status['reasons'])}", "WARNING")
        
        # Trigger retraining
        result = scheduler.trigger_retraining()
        
        if result['success']:
            send_alert("Retraining completed successfully", "INFO")
            logger.info("Retraining completed successfully")
        else:
            send_alert(f"Retraining failed: {result['message']}", "ERROR")
            logger.error(f"Retraining failed: {result['message']}")
    else:
        logger.info("No retraining needed at this time")
    
    logger.info("Retraining monitoring check completed")

if __name__ == "__main__":
    main()
