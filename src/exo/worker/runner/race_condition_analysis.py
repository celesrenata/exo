"""
Race Condition Analysis and Diagnostic Tools

This module provides tools to analyze and reproduce the multiprocessing race condition
that occurs during runner shutdown in multi-node EXO instances.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from loguru import logger


class RaceConditionType(Enum):
    QUEUE_CLOSED_DURING_PUT = "queue_closed_during_put"
    QUEUE_CLOSED_DURING_GET = "queue_closed_during_get"
    RESOURCE_CLOSED_DURING_ACCESS = "resource_closed_during_access"
    CHANNEL_CLEANUP_RACE = "channel_cleanup_race"


@dataclass
class RaceConditionEvent:
    timestamp: datetime
    event_type: RaceConditionType
    process_id: int
    thread_id: int
    resource_name: str
    operation: str
    error_message: str
    stack_trace: Optional[str] = None


class RaceConditionDetector:
    """Detects and logs race conditions in multiprocessing operations"""

    def __init__(self):
        self.events: List[RaceConditionEvent] = []
        self.lock = threading.Lock()
        self.monitoring = False

    def start_monitoring(self):
        """Start monitoring for race conditions"""
        self.monitoring = True
        logger.info("Race condition monitoring started")

    def stop_monitoring(self):
        """Stop monitoring and return collected events"""
        self.monitoring = False
        logger.info(
            f"Race condition monitoring stopped. Collected {len(self.events)} events"
        )
        return self.events.copy()

    def log_race_condition(
        self,
        event_type: RaceConditionType,
        resource_name: str,
        operation: str,
        error_message: str,
        stack_trace: Optional[str] = None,
    ):
        """Log a detected race condition"""
        if not self.monitoring:
            return

        event = RaceConditionEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            resource_name=resource_name,
            operation=operation,
            error_message=error_message,
            stack_trace=stack_trace,
        )

        with self.lock:
            self.events.append(event)

        logger.warning(
            f"Race condition detected: {event_type.value} on {resource_name} during {operation}"
        )


# Global detector instance
race_detector = RaceConditionDetector()


def instrument_queue_operations():
    """Instrument multiprocessing queue operations to detect race conditions"""
    from multiprocessing.queues import Queue

    # Store original methods
    original_put = Queue.put
    original_get = Queue.get
    original_close = Queue.close

    def instrumented_put(self, obj, block=True, timeout=None):
        try:
            return original_put(self, obj, block, timeout)
        except ValueError as e:
            if "closed" in str(e).lower():
                race_detector.log_race_condition(
                    RaceConditionType.QUEUE_CLOSED_DURING_PUT,
                    f"Queue_{id(self)}",
                    "put",
                    str(e),
                    traceback.format_exc(),
                )
            raise

    def instrumented_get(self, block=True, timeout=None):
        try:
            return original_get(self, block, timeout)
        except ValueError as e:
            if "closed" in str(e).lower():
                race_detector.log_race_condition(
                    RaceConditionType.QUEUE_CLOSED_DURING_GET,
                    f"Queue_{id(self)}",
                    "get",
                    str(e),
                    traceback.format_exc(),
                )
            raise

    def instrumented_close(self):
        logger.debug(f"Closing queue {id(self)} from {threading.current_thread().name}")
        return original_close(self)

    # Monkey patch the methods
    Queue.put = instrumented_put
    Queue.get = instrumented_get
    Queue.close = instrumented_close


def create_race_condition_reproducer():
    """Create a script that reproduces the race condition for testing"""

    reproducer_script = '''
import multiprocessing as mp
import time
import threading
from exo.utils.channels import mp_channel, MpSender, MpReceiver

def worker_process(sender: MpSender, receiver: MpReceiver, worker_id: int):
    """Simulate a runner process that can trigger the race condition"""
    try:
        print("Worker {worker_id} starting")
        
        # Simulate some work
        for i in range(5):
            sender.send(f"Message {i} from worker {worker_id}")
            time.sleep(0.1)
        
        print("Worker {worker_id} finishing")
        
    except Exception as e:
        print("Worker {worker_id} error: {e}")
    finally:
        # This is where the race condition occurs
        try:
            sender.close()
            receiver.close()
            sender.join()
            receiver.join()
        except Exception as e:
            print("Worker {worker_id} cleanup error: {e}")

def reproduce_race_condition():
    """Reproduce the multiprocessing race condition"""
    print("Starting race condition reproduction...")
    
    # Create multiple worker processes to increase chance of race condition
    processes = []
    channels = []
    
    for i in range(4):  # Simulate 4-node setup
        sender, receiver = mp_channel()
        channels.append((sender, receiver))
        
        process = mp.Process(
            target=worker_process,
            args=(sender, receiver, i)
        )
        processes.append(process)
        process.start()
    
    # Let processes run for a bit
    time.sleep(2)
    
    # Trigger shutdown - this is where race conditions occur
    print("Triggering shutdown...")
    for sender, receiver in channels:
        try:
            sender.close()
            receiver.close()
        except Exception as e:
            print("Channel cleanup error: {e}")
    
    # Wait for processes to finish
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            print("Process {process.pid} didn't terminate, killing...")
            process.terminate()
            process.join()
    
    print("Race condition reproduction complete")

if __name__ == "__main__":
    reproduce_race_condition()
'''

    return reproducer_script


def analyze_shutdown_timing():
    """Analyze the timing of shutdown operations to identify race windows"""

    timing_data = {
        "runner_shutdown_start": None,
        "queue_close_start": None,
        "queue_close_complete": None,
        "process_join_start": None,
        "process_join_complete": None,
        "cleanup_complete": None,
    }

    def log_timing_event(event_name: str):
        timing_data[event_name] = time.time()
        logger.debug(f"Timing event: {event_name} at {timing_data[event_name]}")

    return log_timing_event, timing_data


def get_queue_state_info(queue):
    """Get diagnostic information about a multiprocessing queue"""
    try:
        info = {
            "queue_id": id(queue),
            "approximate_size": queue.qsize() if hasattr(queue, "qsize") else "unknown",
            "is_closed": getattr(queue, "_closed", "unknown"),
            "thread_info": threading.current_thread().name,
            "process_info": os.getpid(),
        }
        return info
    except Exception as e:
        return {"error": str(e)}


# Import required modules
import os
import traceback
