# Agentic Underwriting Assistant

A comprehensive AI-powered system for automated underwriting risk assessment that integrates multiple data sources, retrieves relevant documents, runs deterministic checks, and produces explainable risk scores.

## Project Goals

- **Data Integration**: Combine policy and claims information from multiple sources
- **Document Retrieval**: Implement RAG system for relevant document search
- **Risk Assessment**: Build deterministic checks and risk calculators
- **AI Agent**: Create intelligent agent with tools and local LLM
- **MLOps**: Implement experiment tracking and model versioning
- **Explainability**: Provide clear rationale for risk scores

## Project Structure

```
Agentic-Underwriting-Assistant/
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data
│   └── external/          # External data sources
├── src/                   # Source code
│   ├── agents/            # AI agents
│   ├── data/              # Data processing
│   ├── models/            # ML models
│   └── utils/             # Utilities
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test files
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Agentic-Underwriting-Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Run the system**
   ```bash
   python -m src.main
   ```

## 🛠️ Features

- **Multi-source Data Integration**
- **RAG-based Document Retrieval**
- **Deterministic Risk Checks**
- **Local LLM Integration**
- **Explainable Risk Scoring**
- **MLOps Pipeline**
- **Experiment Tracking**

## Risk Scoring

The system provides risk scores based on:
- Financial Health (30%)
- Claims History (25%)
- Policy Compliance (20%)
- External Factors (15%)
- Document Quality (10%)

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Risk scoring weights
- RAG settings
- MLOps configuration

## MLOps

- **Experiment Tracking**: MLflow integration
- **Model Registry**: Version control for models
- **Auto-logging**: Automatic experiment logging
- **Performance Monitoring**: Track model performance over time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.