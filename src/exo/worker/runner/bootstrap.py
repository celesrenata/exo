import os

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    PyTorchIPEXRingInstance,
    TinygradRingInstance,
)
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    # Configure backend-specific environment variables
    if isinstance(bound_instance.instance, TinygradRingInstance):
        # Tinygrad backend configuration
        os.environ["EXO_TINYGRAD_ENABLED"] = "true"

        # Check if GPU mode is forced via environment variables (e.g., from systemd)
        opencl_forced = os.environ.get("OPENCL") == "1"
        gpu_forced = os.environ.get("GPU") == "1"

        if opencl_forced and gpu_forced:
            # Force OpenCL GPU mode without detection
            logger.info(
                "Device selection: GPU with OpenCL runtime (forced via environment)",
                backend_type="tinygrad",
                device_type="GPU",
                runtime="OPENCL",
            )
            os.environ["TINYGRAD_BACKEND"] = "GPU"
            capabilities = None  # Skip capabilities detection
        else:
            # Detect hardware capabilities and configure backend
            from exo.worker.engines.tinygrad.device_config import detect_capabilities

            capabilities = detect_capabilities()

            # Set TINYGRAD_BACKEND based on detected device
            if not os.environ.get("TINYGRAD_BACKEND"):
                if capabilities.device_type == "GPU":
                    os.environ["TINYGRAD_BACKEND"] = "GPU"
                elif capabilities.device_type == "METAL":
                    os.environ["TINYGRAD_BACKEND"] = "METAL"
                else:
                    os.environ["TINYGRAD_BACKEND"] = "CPU"

            # Set runtime-specific environment variables
            if capabilities.runtime == "LEVEL_ZERO":
                os.environ["LEVEL_ZERO"] = "1"
                os.environ["GPU"] = "1"
                logger.info(
                    "Device selection: Intel Arc GPU with Level Zero runtime",
                    backend_type="tinygrad",
                    device_type=capabilities.device_type,
                    runtime=capabilities.runtime,
                    device_name=capabilities.device_name,
                    memory_gb=capabilities.memory_gb,
                )
            elif capabilities.runtime == "OPENCL":
                os.environ["OPENCL"] = "1"
                os.environ["GPU"] = "1"
                logger.info(
                    "Device selection: GPU with OpenCL runtime",
                    backend_type=capabilities.device_type,
                    runtime=capabilities.runtime,
                    device_name=capabilities.device_name,
                    memory_gb=capabilities.memory_gb,
                )
            elif capabilities.runtime == "CUDA":
                os.environ["CUDA"] = "1"
                os.environ["GPU"] = "1"
                logger.info(
                    "Device selection: NVIDIA GPU with CUDA runtime",
                    backend_type="tinygrad",
                    device_type=capabilities.device_type,
                    runtime=capabilities.runtime,
                    device_name=capabilities.device_name,
                    memory_gb=capabilities.memory_gb,
                )
            elif capabilities.runtime == "METAL":
                os.environ["METAL"] = "1"
                logger.info(
                    "Device selection: Apple Metal GPU",
                    backend_type="tinygrad",
                    device_type=capabilities.device_type,
                    runtime=capabilities.runtime,
                    device_name=capabilities.device_name,
                    memory_gb=capabilities.memory_gb,
                )
            else:
                # CPU fallback
                logger.info(
                    "Device selection: CPU (no GPU runtime available)",
                    backend_type="tinygrad",
                    device_type=capabilities.device_type,
                    device_name=capabilities.device_name,
                    memory_gb=capabilities.memory_gb,
                    compute_units=capabilities.compute_units,
                )

        logger.info(f"Tinygrad backend: {os.environ.get('TINYGRAD_BACKEND')}")
    elif isinstance(bound_instance.instance, PyTorchIPEXRingInstance):
        # PyTorch+IPEX backend configuration
        os.environ["EXO_PYTORCH_IPEX_ENABLED"] = "true"
        
        # Set PyTorch+IPEX environment variables for optimal performance
        os.environ["PYTORCH_ENABLE_XPU"] = "1"
        os.environ["IPEX_TILE_AS_DEVICE"] = "1"
        
        # Detect Intel Arc GPU and configure device
        try:
            # Lazy import to avoid loading PyTorch unless needed
            import torch
            import intel_extension_for_pytorch as ipex
            
            if torch.xpu.is_available():
                device_count = torch.xpu.device_count()
                if device_count > 0:
                    # Get device properties for logging
                    props = torch.xpu.get_device_properties(0)
                    device_name = props.name if hasattr(props, 'name') else "Intel XPU"
                    total_memory_gb = props.total_memory / (1024**3) if hasattr(props, 'total_memory') else 0
                    
                    logger.info(
                        "Device selection: Intel Arc GPU with PyTorch+IPEX",
                        backend_type="pytorch_ipex",
                        device_type="XPU",
                        device_count=device_count,
                        device_name=device_name,
                        memory_gb=f"{total_memory_gb:.2f}",
                    )
                else:
                    logger.warning(
                        "PyTorch XPU available but no devices found, will fall back to CPU",
                        backend_type="pytorch_ipex",
                    )
            else:
                logger.warning(
                    "Intel XPU not available, PyTorch+IPEX will fall back to CPU",
                    backend_type="pytorch_ipex",
                )
        except ImportError as e:
            logger.warning(
                f"Failed to import PyTorch or IPEX: {e}. Backend will attempt initialization anyway.",
                backend_type="pytorch_ipex",
            )
        except Exception as e:
            logger.warning(
                f"Failed to detect Intel Arc GPU: {e}. Backend will attempt initialization anyway.",
                backend_type="pytorch_ipex",
            )
    else:
        # MLX backend configuration
        fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
        if fast_synch_override == "on" or (
            fast_synch_override != "off"
            and (
                isinstance(bound_instance.instance, MlxJacclInstance)
                and len(bound_instance.instance.jaccl_devices) >= 2
            )
        ):
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"
        else:
            os.environ["MLX_METAL_FAST_SYNCH"] = "0"
        logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        from exo.worker.runner.runner import main

        main(bound_instance, event_sender, task_receiver)
    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
