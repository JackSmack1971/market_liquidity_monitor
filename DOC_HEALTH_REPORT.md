# Documentation Health Report

**Generated**: 2026-01-03  
**Project**: Market Liquidity Monitor  
**Audit Scope**: 7-day code evolution vs. documentation

---

## Executive Summary

**Rot Severity Score**: **MEDIUM** 🟡

The documentation is generally accurate but **significantly incomplete**. Recent major features (multi-exchange comparison, historical backtesting, Docker deployment) are not documented in the main README or QUICKSTART guides.

---

## 1. The "Lying" Index

### Critical Inaccuracies

#### ❌ **README.md Line 100**: Incorrect Frontend Entry Point

```markdown
streamlit run market_liquidity_monitor/frontend/enhanced_app.py
```

**Reality**: The correct entry point is `frontend/app.py`, not `enhanced_app.py`.  
**Impact**: Users will get a file not found error.

#### ❌ **QUICKSTART.md Line 62**: Incorrect Module Path

```bash
streamlit run market_liquidity_monitor/frontend/app.py
```

**Reality**: Should be `streamlit run frontend/app.py` (no `market_liquidity_monitor` prefix when running from project root).  
**Impact**: Import errors for users.

#### ❌ **README.md Lines 57-62**: Incomplete Feature List

The "Advanced Features" section mentions:

- Historical Analysis
- Liquidity Alerts
- Market Impact
- System Health

**Missing**:

- ✗ Multi-Exchange Comparison (NEW - added in latest commit)
- ✗ Historical Backtesting (NEW - added in latest commit)
- ✗ Arbitrage Detection (NEW - added in latest commit)
- ✗ Synthetic Order Book Reconstruction (NEW - added in latest commit)

---

## 2. Semantic Drift Analysis

### Vocabulary Drift

| Documentation Term | Actual Code Term | Status |
|-------------------|------------------|--------|
| "enhanced_app.py" | `app.py` | ❌ Outdated |
| "LiquidityAnalysis" | Still valid | ✅ Current |
| "OrderBook" | Still valid | ✅ Current |
| N/A | `VenueAnalysis` | ⚠️ Undocumented |
| N/A | `CrossExchangeComparison` | ⚠️ Undocumented |
| N/A | `SyntheticOrderBook` | ⚠️ Undocumented |
| N/A | `BacktestReport` | ⚠️ Undocumented |

### Missing Context: New Tools

The following agent tools exist in `agents/tools.py` but are **not documented** anywhere:

1. **`run_historical_backtest`** (Lines 498-656)
   - Purpose: Time-travel execution simulation
   - Parameters: symbol, order_size, side, timeframe, lookback_days
   - Returns: `BacktestReport`
   - **Impact**: Major feature completely undocumented

2. **`compare_exchanges`** (Enhanced - Lines 92-200)
   - NEW: Now supports `order_size` parameter for market impact analysis
   - NEW: Returns `CrossExchangeComparison` with arbitrage detection
   - **Documentation shows**: Basic order book comparison only

3. **`calculate_market_impact`** (Lines 399-495)
   - Enhanced with circuit breaker checks
   - Enhanced with ModelRetry guidance
   - **Documentation**: Not mentioned in QUICKSTART

### Dead Paths

#### ✅ No Dead Paths Found

All referenced configuration files and scripts exist:

- `.env.example` ✅
- `requirements.txt` ✅
- `tests/requirements-test.txt` ✅

---

## 3. Missing Features in Documentation

### Critical Omissions

#### 🆕 **Multi-Exchange Comparison**

**Code Location**: `data_engine/analytics.py:180-354`  
**Models**: `VenueAnalysis`, `CrossExchangeComparison`  
**Documentation**: **NONE**

**What Users Are Missing**:

- How to compare liquidity across multiple exchanges
- How arbitrage opportunities are detected
- Fee-adjusted profit calculations
- Circuit breaker exclusion logic

#### 🆕 **Historical Backtesting**

**Code Location**: `data_engine/analytics.py:357-540`  
**Models**: `SyntheticOrderBook`, `BacktestReport`  
**Tool**: `run_historical_backtest`  
**Documentation**: **NONE**

**What Users Are Missing**:

- How to simulate historical order execution
- Synthetic order book reconstruction methodology
- Volatility-based spread estimation
- How to identify optimal execution windows

#### 🆕 **Docker Deployment**

**Files**: `Dockerfile`, `docker-compose.yml`, `DOCKER_DEPLOYMENT.md`  
**README Reference**: **NONE**

**What Users Are Missing**:

- Production deployment instructions
- Multi-service orchestration
- Security hardening steps
- Scaling strategies

#### 🆕 **Logfire Instrumentation**

**Code Location**: Throughout `agents/tools.py`  
**Documentation**: **NONE**

**What Users Are Missing**:

- How to enable observability
- What metrics are tracked
- How to view performance spans

---

## 4. Context Rot Validation

### Architectural Decision Drift

#### ⚠️ **README Architecture Diagram** (Lines 7-29)

**Current Diagram Shows**:

```
User Interface → Backend → [Data Engine, LLM Agent, Config]
```

**Missing from Diagram**:

- Redis Cache layer
- PostgreSQL Database
- Circuit Breaker pattern
- Multi-exchange parallel execution
- Logfire observability

**Recommendation**: Update architecture diagram to reflect production infrastructure.

---

## 5. Code Examples Validation

### ✅ **API Examples** (QUICKSTART.md Lines 80-104)

All curl examples are **syntactically correct** and reference valid endpoints:

- `/api/v1/orderbook/{symbol}` ✅
- `/api/v1/analyze` ✅
- `/api/v1/quick-check/{symbol}` ✅
- `/api/v1/estimate-slippage` ✅

### ⚠️ **Missing Examples**

No examples exist for:

- Multi-exchange comparison API calls
- Historical backtest requests
- Docker deployment commands (except in DOCKER_DEPLOYMENT.md)

---

## 6. Dependency Drift

### ✅ **requirements.txt** - Recently Updated

All dependencies are **pinned** (latest commit):

- `ccxt==4.5.30` ✅
- `fastapi==0.109.0` ✅
- `streamlit==1.52.2` ✅
- `pydantic-ai==0.0.13` ✅
- `logfire[fastapi]==0.46.0` ✅

**No drift detected** - requirements match code imports.

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Total Documentation Files** | 6 (README, QUICKSTART, DOCKER_DEPLOYMENT, etc.) |
| **Lying Examples** | 2 (file paths) |
| **Undocumented Features** | 4 (multi-exchange, backtest, Docker, Logfire) |
| **Drifted Terms** | 1 ("enhanced_app.py") |
| **Dead Paths** | 0 |
| **Outdated Code Examples** | 0 |

---

## Recommendations

### Priority 1: Fix "Lying" Paths (Surgical)

- Update README.md line 100: `enhanced_app.py` → `app.py`
- Update QUICKSTART.md line 62: Remove `market_liquidity_monitor/` prefix

### Priority 2: Document New Features (Strategic)

- Add "Multi-Exchange Comparison" section to README
- Add "Historical Backtesting" section to README
- Add "Docker Deployment" reference in README (link to DOCKER_DEPLOYMENT.md)
- Update QUICKSTART with new tool examples

### Priority 3: Update Architecture Diagram

- Include Redis, PostgreSQL, Circuit Breaker
- Show Logfire observability layer
- Illustrate multi-exchange parallel execution

---

**Next Steps**: See `REMEDIATION_PLAN.md` for implementation strategy.
