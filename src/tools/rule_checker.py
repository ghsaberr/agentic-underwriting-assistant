"""
Rule Checker Tool - Deterministic Tool B
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RuleChecker:
    """
    Business rules checker that returns binary flags and suggested actions
    """
    
    def __init__(self):
        self.rules = {
            'age_rules': {
                'min_age': 18,
                'max_age': 80,
                'senior_age_threshold': 65,
                'young_age_threshold': 25
            },
            'credit_rules': {
                'excellent_credit': 750,
                'good_credit': 700,
                'fair_credit': 600,
                'poor_credit': 500
            },
            'claims_rules': {
                'max_claims_per_year': 2,
                'high_severity_threshold': 1,
                'recent_claim_days': 365
            },
            'income_rules': {
                'min_income_ratio': 0.1,
                'min_income_absolute': 20000
            },
            'policy_rules': {
                'max_coverage_ratio': 0.8,
                'min_policy_age_days': 30
            }
        }
    
    def check_all_rules(self, policyholder_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check all business rules for policyholder compliance
        
        Args:
            policyholder_data: Dictionary containing policyholder information
            
        Returns:
            Dict containing flags and actions
        """
        try:
            result = {
                'flags': [],
                'actions': [],
                'compliance_score': 100,
                'rule_violations': []
            }
            
            # Check age rules
            age_result = self.check_age_rules(policyholder_data.get('age', 0))
            result['flags'].extend(age_result['flags'])
            result['actions'].extend(age_result['actions'])
            result['rule_violations'].extend(age_result['violations'])
            
            # Check credit rules
            credit_result = self.check_credit_rules(policyholder_data.get('credit_score', 0))
            result['flags'].extend(credit_result['flags'])
            result['actions'].extend(credit_result['actions'])
            result['rule_violations'].extend(credit_result['violations'])
            
            # Check claims rules
            claims_result = self.check_claims_rules(
                policyholder_data.get('claims_history', []),
                policyholder_data.get('policy_start_date', None)
            )
            result['flags'].extend(claims_result['flags'])
            result['actions'].extend(claims_result['actions'])
            result['rule_violations'].extend(claims_result['violations'])
            
            # Check income rules
            income_result = self.check_income_rules(
                policyholder_data.get('annual_income', 0),
                policyholder_data.get('premium_amount', 0)
            )
            result['flags'].extend(income_result['flags'])
            result['actions'].extend(income_result['actions'])
            result['rule_violations'].extend(income_result['violations'])
            
            # Calculate compliance score
            total_violations = len(result['rule_violations'])
            result['compliance_score'] = max(0, 100 - (total_violations * 10))
            
            logger.info(f"Rule check completed: {total_violations} violations, compliance: {result['compliance_score']}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking rules: {e}")
            return {
                'flags': ['rule_check_error'],
                'actions': ['manual_review_required'],
                'compliance_score': 50,
                'rule_violations': ['system_error']
            }
    
    def check_age_rules(self, age: int) -> Dict[str, Any]:
        """Check age-related business rules"""
        result = {'flags': [], 'actions': [], 'violations': []}
        
        if age < self.rules['age_rules']['min_age']:
            result['flags'].append('underage_applicant')
            result['actions'].append('reject_application')
            result['violations'].append('age_below_minimum')
        
        if age > self.rules['age_rules']['max_age']:
            result['flags'].append('overage_applicant')
            result['actions'].append('senior_underwriting_review')
            result['violations'].append('age_above_maximum')
        
        if age >= self.rules['age_rules']['senior_age_threshold']:
            result['flags'].append('senior_applicant')
            result['actions'].append('senior_underwriting_review')
        
        if age <= self.rules['age_rules']['young_age_threshold']:
            result['flags'].append('young_applicant')
            result['actions'].append('young_driver_review')
        
        return result
    
    def check_credit_rules(self, credit_score: int) -> Dict[str, Any]:
        """Check credit score-related business rules"""
        result = {'flags': [], 'actions': [], 'violations': []}
        
        if credit_score < self.rules['credit_rules']['poor_credit']:
            result['flags'].append('poor_credit')
            result['actions'].append('high_premium_required')
            result['violations'].append('credit_below_threshold')
        
        elif credit_score < self.rules['credit_rules']['fair_credit']:
            result['flags'].append('fair_credit')
            result['actions'].append('standard_underwriting')
        
        elif credit_score < self.rules['credit_rules']['good_credit']:
            result['flags'].append('good_credit')
            result['actions'].append('preferred_underwriting')
        
        else:
            result['flags'].append('excellent_credit')
            result['actions'].append('preferred_underwriting')
        
        return result
    
    def check_claims_rules(self, claims_history: List[Dict], policy_start_date: str = None) -> Dict[str, Any]:
        """Check claims history-related business rules"""
        result = {'flags': [], 'actions': [], 'violations': []}
        
        if not claims_history:
            result['flags'].append('no_claims_history')
            result['actions'].append('standard_underwriting')
            return result
        
        # Count recent claims
        recent_claims = 0
        high_severity_claims = 0
        
        for claim in claims_history:
            # Check if claim is recent (within 365 days)
            if policy_start_date:
                try:
                    claim_date = datetime.strptime(claim.get('date', ''), '%Y-%m-%d')
                    policy_date = datetime.strptime(policy_start_date, '%Y-%m-%d')
                    days_diff = (policy_date - claim_date).days
                    
                    if days_diff <= self.rules['claims_rules']['recent_claim_days']:
                        recent_claims += 1
                except:
                    pass
            
            # Check claim severity
            claim_amount = claim.get('amount', 0)
            if claim_amount > 10000:  # High severity threshold
                high_severity_claims += 1
        
        # Apply rules
        if recent_claims > self.rules['claims_rules']['max_claims_per_year']:
            result['flags'].append('excessive_recent_claims')
            result['actions'].append('high_risk_underwriting')
            result['violations'].append('too_many_recent_claims')
        
        if high_severity_claims >= self.rules['claims_rules']['high_severity_threshold']:
            result['flags'].append('high_severity_claims')
            result['actions'].append('specialist_review_required')
            result['violations'].append('high_severity_claims_detected')
        
        if recent_claims > 0:
            result['flags'].append('recent_claims_present')
            result['actions'].append('claims_history_review')
        
        return result
    
    def check_income_rules(self, annual_income: float, premium_amount: float) -> Dict[str, Any]:
        """Check income-related business rules"""
        result = {'flags': [], 'actions': [], 'violations': []}
        
        # Check minimum income
        if annual_income < self.rules['income_rules']['min_income_absolute']:
            result['flags'].append('low_income')
            result['actions'].append('income_verification_required')
            result['violations'].append('income_below_minimum')
        
        # Check income to premium ratio
        if premium_amount > 0:
            income_ratio = annual_income / premium_amount
            if income_ratio < self.rules['income_rules']['min_income_ratio']:
                result['flags'].append('high_premium_ratio')
                result['actions'].append('premium_adjustment_required')
                result['violations'].append('premium_too_high_for_income')
        
        # Income categories
        if annual_income >= 100000:
            result['flags'].append('high_income')
            result['actions'].append('preferred_underwriting')
        elif annual_income >= 50000:
            result['flags'].append('medium_income')
            result['actions'].append('standard_underwriting')
        else:
            result['flags'].append('low_income')
            result['actions'].append('basic_underwriting')
        
        return result
