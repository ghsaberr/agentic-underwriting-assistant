# Retraining/Reindexing Guide

## Overview

This document describes the simple retraining and reindexing system for the Underwriting Agent.

## Components

### 1. RetrainingScheduler (`src/mlops/retraining_scheduler.py`)

Main scheduler that checks if retraining is needed and triggers the process.

**Triggers:**
- **Time-based**: Monthly (30 days)
- **Performance drift**: 15% performance drop
- **Data volume**: New data available

**Usage:**
```python
from src.mlops.retraining_scheduler import RetrainingScheduler

scheduler = RetrainingScheduler()
status = scheduler.check_retraining_needed()
if status['retraining_needed']:
    result = scheduler.trigger_retraining()
```

### 2. API Endpoints

- `GET /retraining-status` - Check if retraining is needed
- `POST /trigger-retraining` - Manually trigger retraining
- `POST /reindex-documents` - Reindex vector store

### 3. Monitoring Script (`scripts/monitor_retraining.py`)

Cron job script for automated monitoring.

## Setup

### 1. Create logs directory
```bash
mkdir -p logs
```

### 2. Set up cron job
```bash
# Edit crontab
crontab -e

# Add this line to run every Monday at 9 AM
0 9 * * 1 cd /path/to/project && python scripts/monitor_retraining.py
```

### 3. Manual monitoring
```bash
# Check retraining status
curl http://localhost:8000/retraining-status

# Trigger retraining manually
curl -X POST http://localhost:8000/trigger-retraining
```

## Configuration

Edit `config/config.yaml` to adjust triggers:

```yaml
retraining:
  time_based_days: 30          # Monthly retraining
  data_drift_threshold: 0.15   # 15% performance drop
  min_samples: 100             # Minimum new samples
```

## Monitoring

### Performance Metrics

The system tracks:
- Accuracy
- Precision
- Recall
- F1 Score

### Alerts

Alerts are sent when:
- API is not responding
- Retraining is needed
- Retraining fails
- Performance drops significantly

### Logs

Check logs in:
- `logs/retraining_monitor.log` - Monitoring logs
- `mlruns/performance_history.json` - Performance history
- `mlruns/retraining_state.json` - Retraining state

## Production Considerations

### 1. Database Integration
- Store performance metrics in database
- Use proper backup strategies
- Implement rollback mechanisms

### 2. Alerting
- Integrate with Slack/Teams
- Email notifications
- PagerDuty for critical alerts

### 3. Scaling
- Use distributed training for large datasets
- Implement A/B testing for new models
- Gradual rollout of new models

### 4. Security
- Secure API endpoints
- Encrypt sensitive data
- Audit trail for all changes

## Troubleshooting

### Common Issues

1. **API not responding**
   - Check if containers are running
   - Verify port 8000 is accessible

2. **Retraining fails**
   - Check logs for error messages
   - Verify data files exist
   - Ensure sufficient disk space

3. **Performance drift not detected**
   - Check if enough data is available
   - Verify MLflow tracking is working
   - Adjust drift threshold if needed

### Debug Commands

```bash
# Check API health
curl http://localhost:8000/health

# Check MLflow status
curl http://localhost:8000/mlflow-summary

# View recent logs
tail -f logs/retraining_monitor.log
```

## Future Enhancements

1. **Advanced Drift Detection**
   - Statistical tests for data drift
   - Model performance monitoring
   - Automated feature importance tracking

2. **Automated Retraining**
   - CI/CD pipeline integration
   - Automated model validation
   - Canary deployments

3. **Advanced Monitoring**
   - Real-time dashboards
   - Anomaly detection
   - Predictive maintenance
