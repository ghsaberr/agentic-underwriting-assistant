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
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=30)
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
        logger.info("Starting risk assessment pipeline...")
        
        try:
            # Step 1: Run deterministic tools
            logger.info("Step 1: Running deterministic tools...")
            tool_outputs = self._run_deterministic_tools(policyholder_data)
            logger.info(f"Tool outputs: {list(tool_outputs.keys())}")
            
            # Step 2: Retrieve relevant documents
            logger.info("Step 2: Retrieving documents...")
            retrieved_docs = self._retrieve_documents(policyholder_data)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 3: Generate LLM response
            logger.info("Step 3: Generating LLM response...")
            llm_response = self._generate_llm_response(
                policyholder_data, retrieved_docs, tool_outputs
            )
            logger.info(f"LLM response keys: {list(llm_response.keys())}")
            
            # Step 4: Combine results with normalization
            logger.info("Step 4: Combining results...")
            # Prefer explicit numeric in rationale: "Risk Score: X"
            rationale_text = str(llm_response.get("rationale", ""))
            import re as _re
            rationale_match = _re.search(r"risk\s*score[^:]*:\s*(\d+\.?\d*)", rationale_text, _re.IGNORECASE)
            if rationale_match:
                score = float(rationale_match.group(1))
            else:
                score_raw = llm_response.get("risk_score", 50.0)
                try:
                    score = float(score_raw)
                except Exception:
                    score = 50.0
            # Heuristic scale detection from rationale
            rationale_text = str(llm_response.get("rationale", ""))
            rationale_lc = rationale_text.lower()
            mentions_0_100 = ("0-100" in rationale_lc) or ("out of 100" in rationale_lc)
            mentions_0_1 = ("0-1" in rationale_lc) or ("0 to 1" in rationale_lc) or ("scale of 0-1" in rationale_lc)
            # Normalize: scale to 0-100 only when score clearly in 0-1 or text says 0-1
            if (0.0 <= score <= 1.0) or mentions_0_1:
                score = score * 100.0
            # Clamp to [0, 100]
            if score < 0.0:
                score = 0.0
            if score > 100.0:
                score = 100.0

            # Normalize level from numeric score to enforce consistency
            if score >= 70:
                normalized_level = "High"
            elif score >= 40:
                normalized_level = "Medium"
            else:
                normalized_level = "Low"

            # Consistency guard: combine with deterministic components (max rule)
            risk_calc_score = 0.0
            try:
                risk_calc_score = float(tool_outputs.get("risk_calculator", {}).get("score", 0.0))
            except Exception:
                risk_calc_score = 0.0

            # Rule-derived penalty (hard rules)
            ph_age = float(policyholder_data.get("age", 0) or 0)
            ph_credit = float(policyholder_data.get("credit_score", 0) or 0)
            ph_income = float(policyholder_data.get("annual_income", 0) or 0)

            rule_penalty = 0.0
            if ph_age < 21:
                rule_penalty += 40.0
            if ph_credit < 500:
                rule_penalty += 40.0
            if ph_income < 10000:
                rule_penalty += 20.0

            # Final score = max of sources
            final_score = max(score, risk_calc_score, rule_penalty)
            if final_score > 100.0:
                final_score = 100.0

            if final_score >= 70:
                final_level = "High"
            elif final_score >= 40:
                final_level = "Medium"
            else:
                final_level = "Low"

            tool_list = list(tool_outputs.keys())
            consistency_applied = rule_penalty > score or risk_calc_score > score
            if consistency_applied and "consistency_guard" not in tool_list:
                tool_list.append("consistency_guard")

            # Build rationale with adjustment note when consistency guard applied
            rationale_out = llm_response.get("rationale", "Assessment completed")
            if consistency_applied:
                reasons = []
                if ph_age < 21:
                    reasons.append("age<21")
                if ph_credit < 500:
                    reasons.append("credit<500")
                if ph_income < 10000:
                    reasons.append("income<10k")
                reasons_text = ", ".join(reasons) if reasons else "rules triggered"
                rationale_out = (
                    f"{rationale_out}\n\nNote: Final risk adjusted by hard rules ({reasons_text}). "
                    f"Combined via max(LLM={round(score,1)}, Calc={round(risk_calc_score,1)}, Rules={round(rule_penalty,1)})."
                )

            result = {
                "risk_score": round(final_score, 1),
                "risk_level": final_level,
                "rationale": rationale_out,
                "retrieved_docs": [doc.get("doc_id", "unknown") for doc in retrieved_docs],
                "tool_citations": tool_list,
                "confidence": llm_response.get("confidence", 0.75)
            }
            
            logger.info(f"Final result: risk_score={result['risk_score']}, risk_level={result['risk_level']}, retrieved_docs={len(result['retrieved_docs'])}")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        logger.info("Starting document retrieval...")
        
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []
        
        try:
            # Create search query based on policyholder data
            query = f"underwriting guidelines for {policyholder_data.get('policy_type', 'insurance')} "
            query += f"age {policyholder_data.get('age', 0)} "
            query += f"credit score {policyholder_data.get('credit_score', 0)}"
            
            logger.info(f"Searching for: {query}")
            
            # Search in vector store
            collection = self.vector_store.get_collection("underwriting_documents")
            logger.info(f"Collection retrieved: {collection.name}")
            
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            logger.info(f"Query results: {len(results.get('documents', [[]])[0])} documents found")
            
            # Format results
            documents = []
            if results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    # Get actual doc_id from metadata if available
                    doc_id = f"DOC_{i+1:03d}"
                    if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]):
                        doc_id = results['metadatas'][0][i].get('doc_id', doc_id)
                    
                    # Get title from metadata
                    title = f"Document {i+1}"
                    if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]):
                        title = results['metadatas'][0][i].get('title', title)
                    
                    documents.append({
                        "doc_id": doc_id,
                        "title": title,
                        "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        "similarity_score": 1 - results['distances'][0][i] if results.get('distances') and results['distances'][0] else 0.5
                    })
            
            logger.info(f"Formatted {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
                timeout=600  # Increased timeout to 10 minutes
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
                        
                        logger.info(f"JSON parsed successfully: {parsed_response}")
                        
                        # Ensure risk_score is in 0-100 range
                        risk_score = parsed_response.get('risk_score', 50.0)
                        if risk_score <= 1.0:  # If score is 0-1, convert to 0-100
                            risk_score = risk_score * 100
                        
                        return {
                            "risk_score": risk_score,
                            "risk_level": parsed_response.get('risk_level', 'Medium'),
                            "rationale": parsed_response.get('rationale', response_text),
                            "retrieved_docs": parsed_response.get('document_citations', []),
                            "tool_citations": parsed_response.get('tool_citations', []),
                            "confidence": parsed_response.get('confidence', 0.8)
                        }
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}")
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
        
        logger.info(f"Parsing text response: {text[:200]}...")
        
        # Extract risk score - look for various patterns
        risk_score = 50.0
        
        # Pattern 1: "Risk Score: 2.6" or "Risk Score (0-100): 2.6" or "Risk Score: 60 (out of 100)"
        score_patterns = [
            r'risk[_\s]*score[:\s]*\([^)]*\)[:\s]*(\d+\.?\d*)',
            r'risk[_\s]*score[:\s]*(\d+\.?\d*)[:\s]*\([^)]*\)',
            r'risk[_\s]*score[:\s]*(\d+\.?\d*)',
            r'score[:\s]*(\d+\.?\d*)[:\s]*\([^)]*\)',
            r'score[:\s]*(\d+\.?\d*)'
        ]
        
        for pattern in score_patterns:
            score_match = re.search(pattern, text, re.IGNORECASE)
            if score_match:
                score_value = float(score_match.group(1))
                logger.info(f"Found score pattern '{pattern}': {score_value}")
                
                # If score is 0-1 range, convert to 0-100
                if score_value <= 1.0:
                    risk_score = score_value * 100
                else:
                    risk_score = score_value
                break
        
        # If no score found, look for any decimal number
        if risk_score == 50.0:
            decimal_match = re.search(r'(\d+\.\d+)', text)
            if decimal_match:
                decimal_value = float(decimal_match.group(1))
                logger.info(f"Found decimal number: {decimal_value}")
                if decimal_value <= 1.0:
                    risk_score = decimal_value * 100
                else:
                    risk_score = decimal_value
        
        # Extract risk level - prefer explicit "Risk Level: <value>" capture
        risk_level = "Medium"
        explicit_level_match = re.search(
            r'risk[_\s]*level(?:\s*\([^)]*\))?\s*:\s*(low|medium|high|critical)',
            text,
            re.IGNORECASE,
        )
        if explicit_level_match:
            level_text = explicit_level_match.group(1).lower()
            logger.info(f"Found explicit risk level: {level_text}")
            risk_level = level_text.capitalize()
        else:
            # Secondary patterns (less strict but still scoped to "level:")
            secondary_level_match = re.search(
                r'level\s*:\s*(low|medium|high|critical)', text, re.IGNORECASE
            )
            if secondary_level_match:
                level_text = secondary_level_match.group(1).lower()
                logger.info(f"Found secondary level: {level_text}")
                risk_level = level_text.capitalize()
            else:
                # Final fallback: derive level from numeric score thresholds
                if risk_score >= 70:
                    risk_level = "High"
                elif risk_score >= 40:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
        
        logger.info(f"Parsed result: score={risk_score}, level={risk_level}")
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "rationale": text,  # return full text
            "retrieved_docs": [],
            "tool_citations": [],
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
