# Intel IPEX Engine Test Suite

This directory contains comprehensive unit and integration tests for the Intel Extension for PyTorch (IPEX) engine integration in EXO.

## Test Structure

### Core Test Modules

1. **`test_ipex_engine_detection.py`** - Tests Intel GPU detection and engine selection logic
   - Intel GPU hardware detection
   - IPEX availability checking
   - Engine selection priority
   - Environment validation
   - Error handling for missing drivers/hardware

2. **`test_ipex_model_loading.py`** - Tests IPEX model loading and initialization
   - Model loading with IPEX optimizations
   - Tokenizer wrapper functionality
   - Model quantization detection
   - Memory management during loading
   - Error handling for incompatible models

3. **`test_ipex_inference.py`** - Tests IPEX text generation and streaming output
   - Warmup inference functionality
   - Single-device text generation
   - Distributed inference across multiple Intel GPUs
   - Streaming output generation
   - Memory error handling during inference
   - Performance monitoring integration

4. **`test_ipex_error_handling.py`** - Tests IPEX error handling and fallback mechanisms
   - IPEX-specific error classes
   - Graceful fallback to CPU/Torch engines
   - Memory error recovery
   - Error context collection
   - Health monitoring and recovery

5. **`test_ipex_dashboard_integration.py`** - Tests Intel IPEX UI elements and dashboard integration
   - Engine information display
   - Intel GPU monitoring data
   - System information integration
   - JSON serialization for dashboard
   - Performance metrics formatting

### Support Files

- **`conftest.py`** - Pytest configuration and common fixtures
- **`test_runner.py`** - Convenient test runner script
- **`__init__.py`** - Package initialization

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From the test directory
python test_runner.py

# Or using pytest directly
pytest test_ipex/ -v
```

### Run Specific Test Modules

```bash
# Run engine detection tests
python test_runner.py test_ipex_engine_detection

# Run inference tests
python test_runner.py test_ipex_inference

# Run dashboard integration tests
python test_runner.py test_ipex_dashboard_integration
```

### Run with Coverage

```bash
pytest test_ipex/ --cov=exo.worker.engines.ipex --cov-report=html
```

### List Available Tests

```bash
python test_runner.py list
```

## Test Categories

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests across components
- `@pytest.mark.slow` - Tests that may take longer to run
- `@pytest.mark.gpu_required` - Tests requiring actual Intel GPU hardware

### Run Tests by Category

```bash
# Run only unit tests
pytest test_ipex/ -m unit

# Run only integration tests
pytest test_ipex/ -m integration

# Skip slow tests
pytest test_ipex/ -m "not slow"
```

## Test Coverage

The test suite covers:

### Engine Detection (Requirements 1.1, 1.2, 3.1, 3.2, 4.1)
- ✅ Intel GPU hardware detection
- ✅ IPEX library availability checking
- ✅ Engine selection logic and priority
- ✅ Environment validation
- ✅ Fallback mechanisms

### Model Loading and Initialization (Requirements 4.2, 4.3, 4.4)
- ✅ IPEX model loading and optimization
- ✅ Tokenizer wrapper functionality
- ✅ Quantization detection and handling
- ✅ Memory management
- ✅ Error handling for incompatible models

### Inference and Generation (Requirements 7.1, 7.3)
- ✅ Single-device text generation
- ✅ Distributed inference across multiple Intel GPUs
- ✅ Streaming output generation
- ✅ Warmup inference
- ✅ Memory error handling
- ✅ Performance monitoring

### Dashboard Integration (Requirements 2.1, 2.2, 2.3, 2.4, 2.5)
- ✅ Engine information display
- ✅ Intel GPU monitoring and metrics
- ✅ System information integration
- ✅ UI element rendering
- ✅ Instance creation and management

### Error Handling and Fallback
- ✅ IPEX-specific error classes
- ✅ Graceful fallback to other engines
- ✅ Memory error recovery
- ✅ Health monitoring and alerts
- ✅ Error context collection

## Mock Strategy

The tests use comprehensive mocking to:

1. **Simulate Intel GPU Hardware** - Mock `torch.xpu` to simulate Intel GPU availability
2. **Mock IPEX Library** - Mock `intel_extension_for_pytorch` for testing without actual IPEX
3. **Mock Transformers** - Mock HuggingFace transformers for model loading tests
4. **Mock Distributed Operations** - Mock `torch.distributed` for distributed inference tests
5. **Mock System Resources** - Mock memory usage, device properties, and health monitoring

## Test Data

Tests use realistic test data including:

- Sample model IDs from the provided list of IPEX-compatible models
- Realistic memory usage patterns
- Typical chat completion tasks
- Common error scenarios
- Performance metrics

## Continuous Integration

These tests are designed to run in CI environments without requiring actual Intel GPU hardware, using comprehensive mocking to simulate all hardware interactions.

## Contributing

When adding new IPEX functionality:

1. Add corresponding unit tests in the appropriate test module
2. Update integration tests if the change affects cross-component behavior
3. Add dashboard tests if the change affects UI/monitoring
4. Update this README if new test categories are added

## Troubleshooting

### Common Issues

1. **Import Errors** - Ensure the EXO package is in your Python path
2. **Missing Dependencies** - Install pytest and other test dependencies
3. **Mock Failures** - Check that mocks match the actual IPEX API

### Debug Mode

Run tests with verbose output and no capture:
```bash
pytest test_ipex/ -v -s --tb=long
```