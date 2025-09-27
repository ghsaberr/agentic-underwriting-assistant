"""
Real Underwriting Agent implementation
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class UnderwritingAgent:
    """
    Real Underwriting Agent that integrates with:
    - Document retrieval (ChromaDB)
    - Deterministic tools (Risk Calculator, Rule Checker)
    - Local LLM (Ollama)
    """
    
    def __init__(self, 
                 chroma_url: str = "http://localhost:8001",
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "llama2:7b"):
        import os
        self.chroma_url = os.getenv("CHROMA_URL", chroma_url)
        self.ollama_url = os.getenv("OLLAMA_URL", ollama_url)
        self.model_name = model_name
        self.risk_calculator = None
        self.rule_checker = None
        self.vector_store = None
        self.llm = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all agent components"""
        try:
            # Initialize Risk Calculator
            from src.tools.risk_calculator import RiskCalculator
            self.risk_calculator = RiskCalculator()
            logger.info("Risk Calculator initialized")
            
            # Initialize Rule Checker
            from src.tools.rule_checker import RuleChecker
            self.rule_checker = RuleChecker()
            logger.info("Rule Checker initialized")
            
            # Initialize Vector Store (ChromaDB)
            self._initialize_vector_store()
            
            # Initialize LLM
            self._initialize_llm()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Fallback to mock mode
            self._fallback_mode = True
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            import chromadb
            # Use environment variable for ChromaDB URL
            if self.chroma_url.startswith("http://"):
                # Extract host and port from URL
                url_parts = self.chroma_url.replace("http://", "").split(":")
                host = url_parts[0]
                port = int(url_parts[1]) if len(url_parts) > 1 else 8000
                self.vector_store = chromadb.HttpClient(host=host, port=port)
            else:
                self.vector_store = chromadb.HttpClient(host="chroma-db", port=8000)
            
            # Test connection by getting collections
            collections = self.vector_store.list_collections()
            logger.info(f"Vector store initialized: {self.chroma_url}")
            logger.info(f"Available collections: {[c.name for c in collections]}")
        except Exception as e:
            logger.warning(f"Vector store not available: {e}")
            self.vector_store = None
    
    def _initialize_llm(self):
        """Initialize Ollama LLM"""
        try:
            import requests
            # Test Ollama connection
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.llm = OllamaLLM(self.model_name, self.ollama_url)
                logger.info(f"LLM initialized: {self.model_name}")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            self.llm = None
    
    def assess_risk(self, policyholder_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main risk assessment pipeline
        """
        try:
            # Step 1: Run deterministic tools
            tool_outputs = self._run_deterministic_tools(policyholder_data)
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(policyholder_data)
            
            # Step 3: Generate LLM response
            llm_response = self._generate_llm_response(
                policyholder_data, retrieved_docs, tool_outputs
            )
            
            # Step 4: Combine results
            result = {
                "risk_score": llm_response.get("risk_score", 50.0),
                "risk_level": llm_response.get("risk_level", "Medium"),
                "rationale": llm_response.get("rationale", "Assessment completed"),
                "retrieved_docs": [doc.get("doc_id", "unknown") for doc in retrieved_docs],
                "tool_citations": list(tool_outputs.keys()),
                "confidence": llm_response.get("confidence", 0.75)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            # Fallback to simple calculation
            return self._fallback_assessment(policyholder_data)
    
    def _run_deterministic_tools(self, policyholder_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run deterministic tools"""
        tool_outputs = {}
        
        try:
            if self.risk_calculator:
                risk_score = self.risk_calculator.calculate_risk(policyholder_data)
                tool_outputs["risk_calculator"] = {
                    "score": risk_score,
                    "level": "High" if risk_score >= 70 else "Medium" if risk_score >= 40 else "Low"
                }
        except Exception as e:
            logger.warning(f"Risk calculator error: {e}")
        
        try:
            if self.rule_checker:
                rules_result = self.rule_checker.check_all_rules(policyholder_data)
                tool_outputs["rule_checker"] = rules_result
        except Exception as e:
            logger.warning(f"Rule checker error: {e}")
        
        return tool_outputs
    
    def _retrieve_documents(self, policyholder_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector store"""
        if not self.vector_store:
            return []
        
        try:
            # Create search query based on policyholder data
            query = f"underwriting guidelines for {policyholder_data.get('policy_type', 'insurance')} "
            query += f"age {policyholder_data.get('age', 0)} "
            query += f"credit score {policyholder_data.get('credit_score', 0)}"
            
            # Search in vector store
            collection = self.vector_store.get_collection("underwriting_documents")
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        "doc_id": f"DOC_{i+1:03d}",
                        "title": f"Document {i+1}",
                        "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        "similarity_score": results['distances'][0][i] if results['distances'] else 0.5
                    })
            
            return documents
            
        except Exception as e:
            logger.warning(f"Document retrieval error: {e}")
            return []
    
    def _generate_llm_response(self, policyholder_data: Dict[str, Any], 
                              retrieved_docs: List[Dict[str, Any]], 
                              tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM response"""
        if not self.llm:
            return self._fallback_llm_response(policyholder_data, retrieved_docs, tool_outputs)
        
        try:
            # Create prompt
            prompt = self._create_prompt(policyholder_data, retrieved_docs, tool_outputs)
            
            # Call LLM
            response = self.llm.generate_response(prompt, retrieved_docs, tool_outputs)
            return response
            
        except Exception as e:
            logger.warning(f"LLM error: {e}")
            return self._fallback_llm_response(policyholder_data, retrieved_docs, tool_outputs)
    
    def _create_prompt(self, policyholder_data: Dict[str, Any], 
                      retrieved_docs: List[Dict[str, Any]], 
                      tool_outputs: Dict[str, Any]) -> str:
        """Create prompt for LLM"""
        prompt = f"""
        Assess the underwriting risk for this policyholder:
        
        Policyholder Data:
        - ID: {policyholder_data.get('policyholder_id', 'Unknown')}
        - Age: {policyholder_data.get('age', 0)}
        - Income: ${policyholder_data.get('annual_income', 0):,.0f}
        - Credit Score: {policyholder_data.get('credit_score', 0)}
        - Policy Type: {policyholder_data.get('policy_type', 'Unknown')}
        - Claims History: {len(policyholder_data.get('claims_history', []))} claims
        
        Please provide a risk assessment with:
        1. Risk Score (0-100)
        2. Risk Level (Low/Medium/High/Critical)
        3. Key Risk Factors
        4. Detailed Rationale
        5. Confidence Level (0-1)
        
        Format as JSON.
        """
        return prompt
    
    def _fallback_assessment(self, policyholder_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback assessment when components fail"""
        # Simple risk calculation
        risk_score = 0
        
        # Age factor
        age = policyholder_data.get('age', 0)
        if age < 25 or age > 65:
            risk_score += 20
        elif age < 30:
            risk_score += 10
        
        # Credit score factor
        credit_score = policyholder_data.get('credit_score', 0)
        if credit_score < 500:
            risk_score += 30
        elif credit_score < 600:
            risk_score += 20
        elif credit_score < 700:
            risk_score += 10
        
        # Income factor
        income = policyholder_data.get('annual_income', 0)
        if income < 30000:
            risk_score += 15
        elif income < 50000:
            risk_score += 8
        
        # Claims factor
        claims = policyholder_data.get('claims_history', [])
        risk_score += len(claims) * 5
        
        # Determine risk level (lower score = lower risk)
        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "rationale": f"Fallback assessment based on age ({age}), credit score ({credit_score}), income (${income:,.0f}), and claims history ({len(claims)} claims)",
            "retrieved_docs": [],
            "tool_citations": ["fallback_calculator"],
            "confidence": 0.6
        }
    
    def _fallback_llm_response(self, policyholder_data: Dict[str, Any], 
                              retrieved_docs: List[Dict[str, Any]], 
                              tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback LLM response when LLM is not available"""
        return {
            "risk_score": 50.0,
            "risk_level": "Medium",
            "rationale": "Assessment completed using available data (LLM not available)",
            "retrieved_docs": [doc.get("doc_id", "unknown") for doc in retrieved_docs],
            "tool_citations": list(tool_outputs.keys()),
            "confidence": 0.6
        }


class OllamaLLM:
    """Ollama LLM integration"""
    
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_response(self, prompt: str, context_docs: List[Dict], tool_outputs: Dict) -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            import requests
            
            # Prepare context
            context_text = ""
            if context_docs:
                context_text = "\n\nRetrieved Documents:\n"
                for i, doc in enumerate(context_docs, 1):
                    context_text += f"{i}. {doc.get('title', 'Document')} (Score: {doc.get('similarity_score', 0):.3f})\n"
                    context_text += f"   Content: {doc.get('content', '')[:200]}...\n\n"
            
            tools_text = ""
            if tool_outputs:
                tools_text = "\n\nTool Outputs:\n"
                for tool_name, tool_data in tool_outputs.items():
                    tools_text += f"- {tool_name}: {tool_data}\n"
            
            full_prompt = f"{prompt}\n{context_text}\n{tools_text}"
            
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1000
                    }
                },
                timeout=120  # Increased timeout to 2 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Try to parse JSON response
                try:
                    import json
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed_response = json.loads(json_str)
                        return parsed_response
                except:
                    pass
                
                # Fallback: extract information from text
                return self._parse_text_response(response_text)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._get_fallback_response()
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails"""
        import re
        
        # Extract risk score
        score_match = re.search(r'risk[_\s]*score[:\s]*(\d+)', text, re.IGNORECASE)
        risk_score = float(score_match.group(1)) if score_match else 50.0
        
        # Extract risk level
        risk_level = "Medium"
        if 'high' in text.lower() or 'critical' in text.lower():
            risk_level = "High"
        elif 'low' in text.lower():
            risk_level = "Low"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "rationale": text[:500] + "..." if len(text) > 500 else text,
            "confidence": 0.8
        }
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Fallback response when Ollama fails"""
        return {
            "risk_score": 50.0,
            "risk_level": "Medium",
            "rationale": "Assessment completed using available data",
            "confidence": 0.6
        }
