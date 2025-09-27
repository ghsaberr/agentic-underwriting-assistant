"""
Risk Calculator Tool - Deterministic Tool A
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskCalculator:
    """
    Deterministic risk calculator that computes numeric risk proxy
    """
    
    def __init__(self):
        self.weights = {
            'age': 0.2,
            'credit_score': 0.3,
            'income': 0.2,
            'claims_history': 0.3
        }
    
    def calculate_risk(self, policyholder_data: Dict[str, Any]) -> float:
        """
        Calculate risk score based on policyholder data
        
        Args:
            policyholder_data: Dictionary containing policyholder information
            
        Returns:
            float: Risk score between 0-100
        """
        try:
            risk_score = 0
            
            # Age factor (0-20 points)
            age = policyholder_data.get('age', 0)
            if age < 25 or age > 65:
                age_risk = 20
            elif age < 30:
                age_risk = 15
            elif age < 40:
                age_risk = 5
            else:
                age_risk = 0
            
            risk_score += age_risk * self.weights['age']
            
            # Credit score factor (0-30 points)
            credit_score = policyholder_data.get('credit_score', 0)
            if credit_score < 500:
                credit_risk = 30
            elif credit_score < 600:
                credit_risk = 25
            elif credit_score < 700:
                credit_risk = 15
            elif credit_score < 750:
                credit_risk = 5
            else:
                credit_risk = 0
            
            risk_score += credit_risk * self.weights['credit_score']
            
            # Income factor (0-20 points)
            income = policyholder_data.get('annual_income', 0)
            if income < 30000:
                income_risk = 20
            elif income < 50000:
                income_risk = 15
            elif income < 75000:
                income_risk = 8
            elif income < 100000:
                income_risk = 3
            else:
                income_risk = 0
            
            risk_score += income_risk * self.weights['income']
            
            # Claims history factor (0-30 points)
            claims_history = policyholder_data.get('claims_history', [])
            claims_count = len(claims_history)
            
            if claims_count >= 5:
                claims_risk = 30
            elif claims_count >= 3:
                claims_risk = 25
            elif claims_count >= 2:
                claims_risk = 15
            elif claims_count >= 1:
                claims_risk = 8
            else:
                claims_risk = 0
            
            risk_score += claims_risk * self.weights['claims_history']
            
            # Ensure score is between 0-100
            risk_score = max(0, min(100, risk_score))
            
            logger.info(f"Risk calculated: {risk_score:.2f} for policyholder {policyholder_data.get('policyholder_id', 'Unknown')}")
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating risk: {e}")
            return 50.0  # Default medium risk
