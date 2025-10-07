# Agentic Underwriting Assistant

A comprehensive AI-powered system for automated underwriting risk assessment that integrates multiple data sources, retrieves relevant documents, runs deterministic checks, and produces explainable risk scores.

## Project Goals

- **Data Integration**: Combine policy and claims information from multiple sources
- **Document Retrieval**: Implement RAG system for relevant document search
- **Risk Assessment**: Build deterministic checks and risk calculators
- **AI Agent**: Create intelligent agent with tools and local LLM
- **MLOps**: Implement experiment tracking and model versioning
- **Explainability**: Provide clear rationale for risk scores


# Test API
```
curl http://localhost:8000/health
```

# Risk assessment
```
curl -X POST http://localhost:8000/assess-risk \
  -H "Content-Type: application/json" \
  -d '{"policyholder_id": "PH_001", "age": 35, "credit_score": 720, "annual_income": 50000}'
```

- **AI Agent**: Ollama + LangChain + RAG
- **Vector Store**: ChromaDB for document retrieval
- **API**: FastAPI with MLflow tracking

## Features

- ✅ **Risk Assessment**: 0-100 risk scores with explanations
- ✅ **Document Retrieval**: RAG-based relevant document search
- ✅ **Deterministic Tools**: Risk calculator & rule checker
- ✅ **Local LLM**: Ollama integration (llama2:7b)
- ✅ **MLOps**: MLflow tracking
- ✅ **API**: RESTful endpoints for integration

## Services
```
| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | Main FastAPI application |
| MLflow UI | 5000 | Experiment tracking |
| ChromaDB | 8001 | Vector database |
| Ollama | 11434 | Local LLM server |
```

## Project Structure
```
├── src/
│   ├── agents/          # AI Agent (RAG + LLM)
│   ├── api/            # FastAPI endpoints
│   ├── tools/          # Risk calculator & rule checker
│   └── mlops/          # MLflow
├── data/               # Data
├── config/             # Configuration files
├── docs/               # Documentation
└── docker-compose.yml  # Multi-container setup
```


## API Endpoints

- `GET /health` - Health check
- `POST /assess-risk` - Risk assessment
- `GET /retraining-status` - Check retraining needs
- `GET /mlflow-ui` - MLflow UI info


## MLflow

- **UI**: http://localhost:5000
- **Tracking**: Automatic experiment logging
- **Artifacts**: Full rationale & model outputs

## Configuration

Edit `config/config.yaml`:
- LLM settings (Ollama)
- Risk scoring thresholds
- RAG parameters

## Documentation

- [API Documentation](http://localhost:8000/docs)
- [MLflow UI](http://localhost:5000)

## Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f underwriting-api

# Stop services
docker-compose down
```

## License

MIT License - see [LICENSE](LICENSE) file.