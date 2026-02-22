# AGENTS.md - Developer Guide for LLM-TradeBot

This file provides guidance for agents working on the LLM-TradeBot codebase.

## Project Overview

LLM-TradeBot is an intelligent multi-agent quantitative trading bot based on the Adversarial Decision Framework (ADF). It uses Python 3.11+ with async/await patterns, multi-agent architecture, and supports backtesting and live trading.

---

## Build, Run, and Test Commands

### Running the Application

```bash
# Local installation (recommended for development)
./install.sh          # One-time setup: creates venv, installs deps, generates .env
./start.sh            # Start dashboard in test mode (localhost:8000)

# With specific modes
./start.sh --test --mode continuous    # Test mode + continuous
./start.sh --mode continuous           # Production mode
./start.sh --test                     # Single run in test mode

# Docker deployment (recommended for production)
cd docker && docker-compose up -d
```

### Running Tests

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_agent_config.py

# Run a single test function
pytest tests/test_agent_config.py::TestAgentConfigDefaults::test_default_enabled_agents -v

# Run tests matching a pattern
pytest -k "test_rsi"

# Run with coverage (if installed)
pytest --cov=src --cov-report=html

# Run tests in a specific directory
pytest tests/agents/
```

### Code Quality Tools

```bash
# Format code (install black first: pip install black)
black src/ tests/

# Lint code (install ruff first: pip install ruff)
ruff check src/ tests/

# Type checking (install mypy first: pip install mypy)
mypy src/
```

---

## Code Style Guidelines

### General Principles

- **Clarity over cleverness**: Write readable code that others can understand
- **Small, focused functions**: Each function should do one thing well
- **Meaningful names**: Use descriptive names for all identifiers
- **No premature optimization**: Optimize only when there's a proven performance need

### Formatting

- **Line length**: Maximum 100 characters (soft limit at 120)
- **Indentation**: 4 spaces (no tabs)
- **Blank lines**: Two blank lines between top-level definitions, one between methods
- **Trailing commas**: Use trailing commas in multi-line collections

### Import Conventions

```python
# Standard library first (alphabetically)
import os
import sys
from typing import Any, Dict, List, Optional

# Third-party packages
import pandas as pd
import numpy as np
import pytest

# Local application imports (absolute imports from src/)
from src.agents.agent_config import AgentConfig
from src.llm.base import BaseLLMClient

# Relative imports within same package (when appropriate)
from . import module_name
from ..parent import ParentClass
```

- Always use absolute imports from `src/` for cross-module imports
- Group imports: stdlib → third-party → local
- Sort each group alphabetically

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `agent_config.py`, `data_saver.py` |
| Classes | PascalCase | `AgentConfig`, `BaseLLMClient` |
| Functions | snake_case | `calculate_rsi()`, `fetch_market_data()` |
| Variables | snake_case | `config`, `initial_capital` |
| Constants | UPPER_SNAKE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private members | leading underscore | `_private_method()`, `_internal_state` |
| Type variables | PascalCase | `InputT`, `OutputT` |

### Type Hints

**Always use type hints for function signatures:**

```python
# Good - explicit types
def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    ...

def process_signal(self, signal_data: Dict[str, Any]) -> Optional[float]:
    """Process trading signal."""
    ...

# Good - generic types
from typing import Dict, List, Optional, Any

async def execute(self, input_data: InputT) -> OutputT:
    """Execute agent with typed input/output."""
    ...

# Good - dataclasses with types
from dataclasses import dataclass

@dataclass
class AgentConfig:
    predict_agent: bool = True
    ai_prediction_filter_agent: bool = True
```

**Avoid:**
- `from typing import *` (import specific types)
- Untyped function signatures
- Using `Any` unless absolutely necessary

### Docstrings

Use Google-style docstrings:

```python
def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price series (typically close prices).
        period: RSI period (default 14).
    
    Returns:
        RSI values as Series with same index as input.
    
    Raises:
        ValueError: If data length is less than period.
    
    Example:
        >>> data = pd.Series([100, 102, 101, 103])
        >>> calculate_rsi(data, period=2)
    """
```

### Error Handling

```python
# Use specific exception types
try:
    result = await client.fetch_data(symbol)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        # Rate limited - retry with backoff
        await asyncio.sleep(backoff_time)
    elif e.response.status_code >= 500:
        # Server error - retry
        raise
    else:
        # Client error - don't retry
        logger.error(f"Client error: {e}")
        raise
except ValueError as e:
    # Expected error - handle gracefully
    logger.warning(f"Invalid configuration: {e}")
    return None
except Exception as e:
    # Catch-all for unexpected errors
    logger.exception(f"Unexpected error in fetch_data: {e}")
    raise
```

**Rules:**
- Never catch `Exception` unless re-raising
- Use specific exception types
- Log errors before re-raising
- Return `None` or sensible defaults when appropriate
- Use custom exceptions for domain-specific errors

### Async/Await Patterns

```python
# Use async for I/O-bound operations
import asyncio

async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Any]:
    """Fetch market data concurrently."""
    tasks = [self._fetch_symbol(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {symbol: r for symbol, r in zip(symbols, results) if not isinstance(r, Exception)}

# Always handle exceptions in gather
async def safe_gather(*tasks):
    """Gather with exception handling."""
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

### Dataclasses for Configuration

Use `@dataclass` for configuration objects:

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

@dataclass
class LLMConfig:
    """LLM configuration."""
    api_key: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 120
    max_retries: int = 5
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("api_key is required")
```

### Abstract Base Classes

Use ABC for interfaces and base classes:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base for all agents."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier (snake_case)."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: InputT) -> OutputT:
        """Execute agent logic."""
        pass
```

---

## Project Structure

```
trade-bot/
├── src/
│   ├── agents/       # Multi-agent system
│   ├── api/          # Exchange API clients
│   ├── backtest/    # Backtesting engine
│   ├── cli/         # Terminal display
│   ├── config/      # Configuration
│   ├── data/        # Data handling
│   ├── exchanges/   # Exchange integrations
│   ├── execution/   # Order execution
│   ├── features/    # Feature engineering
│   ├── llm/         # LLM clients (OpenAI, DeepSeek, Claude, etc.)
│   ├── models/      # ML models
│   ├── monitoring/  # Monitoring/logging
│   ├── risk/        # Risk management
│   ├── server/      # Web dashboard
│   ├── strategy/    # Strategy logic
│   └── utils/       # Utilities
├── tests/           # Unit tests (pytest)
├── config/          # Configuration files
├── web/             # Dashboard frontend
└── docker/          # Docker files
```

---

## Testing Guidelines

### Test Organization

```python
import pytest
import pandas as pd
from src.backtest.agent_wrapper import BacktestSignalCalculator

class TestSignalCalculator:
    """Test BacktestSignalCalculator functionality."""
    
    def test_rsi_no_division_by_zero(self):
        """RSI should handle all-zero changes gracefully."""
        calc = BacktestSignalCalculator()
        data = pd.Series([100.0] * 50)
        
        rsi = calc.calculate_rsi(data)
        
        assert not rsi.isna().all()
        assert not pd.isinf(rsi).any()
    
    def test_rsi_normal_calculation(self):
        """RSI should return values between 0-100."""
        calc = BacktestSignalCalculator()
        data = pd.Series([100, 102, 101, 103, 105] * 10)
        
        rsi = calc.calculate_rsi(data, period=14)
        
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
```

### Test Naming

- Use descriptive test names: `test_<what>_<expected_behavior>`
- Group tests in classes by functionality
- Use docstrings to explain test intent

---

## Configuration

### Environment Variables

Sensitive data goes in `.env` (never commit this):
```
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
DEEPSEEK_API_KEY=your_key
```

### YAML Configuration

Non-sensitive config in `config.yaml`:
- Agent enable/disable flags
- Trading parameters (symbols, timeframes, leverage)
- Risk parameters
- LLM provider settings

---

## Common Pitfalls to Avoid

1. **Don't use relative imports** for cross-module code - use `from src.agents import ...`
2. **Don't catch bare `Exception`** - catch specific types
3. **Don't use `Any` loosely** - use proper type hints
4. **Don't forget `__init__.py`** in new directories
5. **Don't hardcode paths** - use `os.path.join()` or Path objects
6. **Don't block in async code** - use `await` instead of `.result()`
7. **Don't ignore lint warnings** - fix them promptly
8. **Don't commit secrets** - use `.env` and `.gitignore`

---

## Useful Commands Summary

```bash
# Development
pip install -r requirements.txt    # Install dependencies
python main.py --test              # Run in test mode
python main.py --mode continuous    # Production continuous

# Testing  
pytest                              # All tests
pytest path/to/test.py -v          # Single file
pytest -k "pattern"                 # Pattern match

# Code quality
black src/ tests/                   # Format
ruff check src/                    # Lint
mypy src/                          # Type check
```
