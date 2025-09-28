#!/usr/bin/env python3
"""
Demo Script for Agentic Underwriting Assistant
One-command demo with 3 test cases
"""

import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header():
    """Print demo header"""
    print("=" * 80)
    print("AGENTIC UNDERWRITING ASSISTANT - DEMO")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_hardware_info():
    """Print hardware requirements and info"""
    print("HARDWARE REQUIREMENTS:")
    print("-" * 40)
    print("Model: llama2:7b (via Ollama)")
    print("CPU: 4+ cores recommended")
    print("RAM: 8GB minimum, 16GB recommended")
    print("Storage: 5GB for model + data")
    print("Network: Internet for initial model download")
    print("GPU: Optional (CPU-only works fine)")
    print()

def print_demo_cases():
    """Print demo test cases"""
    print("DEMO TEST CASES:")
    print("-" * 40)
    
    cases = [
        {
            "name": "Low Risk Case",
            "description": "Experienced driver with excellent credit",
            "data": {
                "policyholder_id": "DEMO_001",
                "age": 45,
                "credit_score": 780,
                "annual_income": 85000,
                "policy_type": "auto",
                "coverage_amount": 150000,
                "years_at_address": 12,
                "claims_per_year": 0.0,
                "claim_amount_sum": 0,
                "policy_age_days": 1095,
                "claim_to_coverage_ratio": 0.0,
                "premium_to_income_ratio": 0.03
            }
        },
        {
            "name": "Medium Risk Case", 
            "description": "Young professional with good credit",
            "data": {
                "policyholder_id": "DEMO_002",
                "age": 28,
                "credit_score": 680,
                "annual_income": 55000,
                "policy_type": "auto",
                "coverage_amount": 100000,
                "years_at_address": 3,
                "claims_per_year": 0.5,
                "claim_amount_sum": 3000,
                "policy_age_days": 500,
                "claim_to_coverage_ratio": 0.03,
                "premium_to_income_ratio": 0.06
            }
        },
        {
            "name": "High Risk Case",
            "description": "Young driver with poor credit and claims history",
            "data": {
                "policyholder_id": "DEMO_003",
                "age": 18,
                "credit_score": 500,
                "annual_income": 18000,
                "policy_type": "auto",
                "coverage_amount": 200000,
                "years_at_address": 1,
                "claims_per_year": 2.0,
                "claim_amount_sum": 18000,
                "policy_age_days": 180,
                "claim_to_coverage_ratio": 0.09,
                "premium_to_income_ratio": 0.15
            }
        }
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"{i}. {case['name']}")
        print(f"   {case['description']}")
        print(f"   Age: {case['data']['age']}, Credit: {case['data']['credit_score']}, Income: ${case['data']['annual_income']:,}")
        print()
    
    return cases

def run_demo_case(agent, case, case_num):
    """Run a single demo case"""
    print(f"Running {case['name']}...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Run risk assessment
        result = agent.assess_risk(case['data'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        print(f"Assessment completed in {processing_time:.2f} seconds")
        print()
        print(f"RISK ASSESSMENT RESULTS:")
        print(f"   Risk Score: {result['risk_score']}/100")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Retrieved Docs: {len(result['retrieved_docs'])}")
        print(f"   Tools Used: {', '.join(result['tool_citations'])}")
        print()
        print(f"RATIONALE:")
        print(f"   {result['rationale'][:200]}...")
        print()
        
        return {
            "case": case['name'],
            "risk_score": result['risk_score'],
            "risk_level": result['risk_level'],
            "processing_time": processing_time,
            "success": True
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "case": case['name'],
            "error": str(e),
            "success": False
        }

def print_summary(results):
    """Print demo summary"""
    print("DEMO SUMMARY:")
    print("=" * 50)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    print()
    
    if successful:
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        print(f"Average processing time: {avg_time:.2f} seconds")
        print()
        
        print("RISK SCORE DISTRIBUTION:")
        for result in successful:
            print(f"   {result['case']}: {result['risk_score']}/100 ({result['risk_level']})")
        print()
    
    if failed:
        print("FAILED CASES:")
        for result in failed:
            print(f"   {result['case']}: {result['error']}")
        print()

def main():
    """Main demo function"""
    print_header()
    print_hardware_info()
    
    # Check if services are running
    print("CHECKING SERVICES...")
    print("-" * 40)
    
    try:
        import requests
        # Check API health
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("API is running")
        else:
            print("API is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("API is not running. Please start with: docker-compose up -d")
        return
    except Exception as e:
        print(f"Error checking API: {e}")
        return
    
    print()
    
    # Initialize agent
    print("INITIALIZING AGENT...")
    print("-" * 40)
    
    try:
        from agents.underwriting_agent import UnderwritingAgent
        agent = UnderwritingAgent()
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return
    
    print()
    
    # Get demo cases
    cases = print_demo_cases()
    
    # Run demo cases
    print("RUNNING DEMO CASES...")
    print("=" * 50)
    
    results = []
    for i, case in enumerate(cases, 1):
        result = run_demo_case(agent, case, i)
        results.append(result)
        
        if i < len(cases):
            print("Waiting 2 seconds before next case...")
            time.sleep(2)
            print()
    
    # Print summary
    print_summary(results)
    
    print("DEMO COMPLETED!")
    print("=" * 50)
    print("For more information:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - MLflow UI: http://localhost:5000")
    print("   - README: ./README.md")
    print()

if __name__ == "__main__":
    main()
