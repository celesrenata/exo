"""Linux system monitoring utilities."""

import os

from exo.shared.types.profiling import SystemPerformanceProfile

"""Linux system monitoring utilities."""




def get_cpu_temperature() -> float:
    """Get CPU temperature from thermal zones."""
    try:
        # Try psutil first if available
        try:
            import psutil

            temps = psutil.sensors_temperatures()
            if temps:
                # Look for CPU-related sensors
                for sensor_name, entries in temps.items():
                    if any(
                        keyword in sensor_name.lower()
                        for keyword in ["cpu", "core", "processor"]
                    ) and entries:
                        return entries[0].current
                # Fallback to first available sensor
                first_sensor = next(iter(temps.values()))
                if first_sensor:
                    return first_sensor[0].current
        except ImportError:
            pass

        # Fallback to manual thermal zone reading
        thermal_zones = []
        thermal_dir = "/sys/class/thermal"

        if not os.path.exists(thermal_dir):
            return 0.0

        for item in os.listdir(thermal_dir):
            if item.startswith("thermal_zone"):
                thermal_zones.append(item)

        if not thermal_zones:
            return 0.0

        # Read temperature from first available thermal zone
        for zone in thermal_zones:
            temp_file = f"{thermal_dir}/{zone}/temp"
            try:
                with open(temp_file, "r") as f:
                    temp_millicelsius = int(f.read().strip())
                    return temp_millicelsius / 1000.0
            except (FileNotFoundError, ValueError, PermissionError):
                continue

        return 0.0
    except Exception:
        return 0.0


def get_cpu_usage() -> tuple[float, float]:
    """Get CPU usage percentages (performance cores, efficiency cores)."""
    try:
        # Try psutil first if available
        try:
            import psutil

            usage = psutil.cpu_percent(interval=0.1)
            # For simplicity, return same value for both P and E cores
            return usage, usage
        except ImportError:
            pass

        # Fallback to manual /proc/stat reading
        with open("/proc/stat", "r") as f:
            line = f.readline()

        if not line.startswith("cpu "):
            return 0.0, 0.0

        # Parse CPU times: user, nice, system, idle, iowait, irq, softirq, steal
        fields = line.split()[1:]
        if len(fields) < 4:
            return 0.0, 0.0

        idle = int(fields[3])
        total = sum(int(field) for field in fields)

        if total == 0:
            return 0.0, 0.0

        usage = ((total - idle) / total) * 100.0

        # For simplicity, return same value for both P and E cores
        return usage, usage

    except Exception:
        return 0.0, 0.0


def get_system_power() -> float:
    """Get system power consumption if available."""
    try:
        # Try to read power consumption from RAPL (Running Average Power Limit)
        rapl_dir = "/sys/class/powercap"

        if not os.path.exists(rapl_dir):
            return 0.0

        # Look for intel-rapl or similar power domains
        for item in os.listdir(rapl_dir):
            if "intel-rapl" in item:
                energy_file = f"{rapl_dir}/{item}/energy_uj"
                try:
                    with open(energy_file, "r") as f:
                        energy_microjoules = int(f.read().strip())
                        # Convert to watts (this is instantaneous, would need time sampling for real power)
                        return energy_microjoules / 1000000.0  # Convert to joules
                except (FileNotFoundError, ValueError, PermissionError):
                    continue

        return 0.0
    except Exception:
        return 0.0


def get_memory_usage() -> tuple[int, int, int, int]:
    """Get memory usage info (total_ram, available_ram, total_swap, available_swap) in bytes."""
    try:
        # Try psutil first if available
        try:
            import psutil

            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            return vm.total, vm.available, sm.total, sm.free
        except ImportError:
            pass

        # Fallback to manual /proc/meminfo reading
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    # Extract numeric value (remove 'kB' suffix and convert to bytes)
                    value_str = value.strip().split()[0]
                    meminfo[key.strip()] = int(value_str) * 1024  # Convert kB to bytes

        ram_total = meminfo.get("MemTotal", 0)
        ram_available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        swap_total = meminfo.get("SwapTotal", 0)
        swap_free = meminfo.get("SwapFree", 0)

        return ram_total, ram_available, swap_total, swap_free

    except Exception:
        return 0, 0, 0, 0


def get_linux_system_profile() -> SystemPerformanceProfile:
    """Get system performance profile for Linux."""
    try:
        temp = get_cpu_temperature()
        pcpu_usage, ecpu_usage = get_cpu_usage()
        sys_power = get_system_power()

        return SystemPerformanceProfile(
            gpu_usage=0.0,  # GPU monitoring would require additional tools
            temp=temp,
            sys_power=sys_power,
            pcpu_usage=pcpu_usage,
            ecpu_usage=ecpu_usage,
        )
    except Exception:
        # Return default profile if anything fails
        return SystemPerformanceProfile(
            gpu_usage=0.0,
            temp=0.0,
            sys_power=0.0,
            pcpu_usage=0.0,
            ecpu_usage=0.0,
        )
