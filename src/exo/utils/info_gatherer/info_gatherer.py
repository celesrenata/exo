import os
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Self, cast

import anyio
from anyio import create_task_group, open_process
from anyio.abc import TaskGroup
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio.streams.text import TextReceiveStream
from loguru import logger

from exo.shared.constants import EXO_CONFIG_FILE
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    SystemPerformanceProfile,
)
from exo.shared.types.thunderbolt import TBConnection, TBConnectivity, TBIdentifier
from exo.utils.channels import Sender
from exo.utils.pydantic_ext import TaggedModel

from .macmon import MacmonMetrics
from .system_info import get_friendly_name, get_model_and_chip, get_network_interfaces

IS_DARWIN = sys.platform == "darwin"

# Import Linux monitoring for non-macOS systems
if not IS_DARWIN:
    from .linux_monitor import get_linux_system_profile


class StaticNodeInformation(TaggedModel):
    """Node information that should NEVER change, to be gathered once at startup"""

    model: str
    chip: str

    @classmethod
    async def gather(cls) -> Self:
        model, chip = await get_model_and_chip()
        return cls(model=model, chip=chip)


class NodeNetworkInterfaces(TaggedModel):
    ifaces: Sequence[NetworkInterfaceInfo]


class MacTBIdentifiers(TaggedModel):
    idents: Sequence[TBIdentifier]


class MacTBConnections(TaggedModel):
    conns: Sequence[TBConnection]


class NodeConfig(TaggedModel):
    """Node configuration from EXO_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    # TODO
    @classmethod
    async def gather(cls) -> Self | None:
        cfg_file = anyio.Path(EXO_CONFIG_FILE)
        await cfg_file.touch(exist_ok=True)
        async with await cfg_file.open("rb") as f:
            try:
                contents = (await f.read()).decode("utf-8")
                data = tomllib.loads(contents)
                return cls.model_validate(data)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError):
                logger.warning("Invalid config file, skipping...")
                return None


class MiscData(TaggedModel):
    """Node information that may slowly change that doesn't fall into the other categories"""

    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        return cls(friendly_name=await get_friendly_name())


class LinuxSystemMetrics(TaggedModel):
    """Linux system metrics similar to MacmonMetrics"""
    
    system_profile: SystemPerformanceProfile
    memory: MemoryUsage
    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        """Gather Linux system metrics"""
        system_profile = get_linux_system_profile()
        friendly_name = await get_friendly_name()
        
        # Get memory info using basic file reading since psutil might not be available
        try:
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
            
            memory = MemoryUsage.from_bytes(
                ram_total=ram_total,
                ram_available=ram_available,
                swap_total=swap_total,
                swap_available=swap_free,
            )
        except Exception:
            # Fallback to default values if reading fails
            memory = MemoryUsage.from_bytes(
                ram_total=0,
                ram_available=0,
                swap_total=0,
                swap_available=0,
            )
        
        return cls(system_profile=system_profile, memory=memory, friendly_name=friendly_name)


class EngineInformation(TaggedModel):
    """Engine and inference capability information"""
    
    available_engines: list[str]
    selected_engine: str
    mlx_available: bool
    torch_available: bool
    cpu_available: bool

    @classmethod
    async def gather(cls) -> Self:
        from exo.worker.engines.engine_utils import get_engine_info
        
        engine_info = get_engine_info()
        return cls(
            available_engines=engine_info.get("available_engines", []),
            selected_engine=engine_info.get("selected_engine", "unknown"),
            mlx_available=engine_info.get("mlx_available", False),
            torch_available=engine_info.get("torch_available", False),
            cpu_available=engine_info.get("cpu_available", False),
        )


async def _gather_iface_map() -> dict[str, str] | None:
    proc = await anyio.run_process(
        ["networksetup", "-listallhardwareports"], check=False
    )
    if proc.returncode != 0:
        return None

    ports: dict[str, str] = {}
    port = ""
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.startswith("Hardware Port:"):
            port = line.split(": ")[1]
        elif line.startswith("Device:"):
            ports[port] = line.split(": ")[1]
            port = ""
    if "" in ports:
        del ports[""]
    return ports


GatheredInfo = (
    MacmonMetrics
    | LinuxSystemMetrics
    | MemoryUsage
    | NodeNetworkInterfaces
    | MacTBIdentifiers
    | MacTBConnections
    | NodeConfig
    | MiscData
    | StaticNodeInformation
    | EngineInformation
)


@dataclass
class InfoGatherer:
    info_sender: Sender[GatheredInfo]
    interface_watcher_interval: float | None = 10
    misc_poll_interval: float | None = 60
    system_profiler_interval: float | None = 5 if IS_DARWIN else None
    memory_poll_rate: float | None = None if IS_DARWIN else 1
    macmon_interval: float | None = 1 if IS_DARWIN else None
    _tg: TaskGroup = field(init=False, default_factory=create_task_group)

    async def run(self):
        async with self._tg as tg:
            if (macmon_path := shutil.which("macmon")) is not None:
                tg.start_soon(self._monitor_macmon, macmon_path)
            elif not IS_DARWIN:
                # Use Linux system monitoring when macmon is not available
                tg.start_soon(self._monitor_linux_system)
            if IS_DARWIN:
                tg.start_soon(self._monitor_system_profiler)
            tg.start_soon(self._watch_system_info)
            tg.start_soon(self._monitor_memory_usage)
            tg.start_soon(self._monitor_misc)

            nc = await NodeConfig.gather()
            if nc is not None:
                await self.info_sender.send(nc)
            sni = await StaticNodeInformation.gather()
            await self.info_sender.send(sni)
            
            # Gather engine information once at startup
            engine_info = await EngineInformation.gather()
            await self.info_sender.send(engine_info)

    def shutdown(self):
        self._tg.cancel_scope.cancel()

    async def _monitor_misc(self):
        if self.misc_poll_interval is None:
            return
        prev = await MiscData.gather()
        while True:
            curr = await MiscData.gather()
            if prev != curr:
                prev = curr
                await self.info_sender.send(curr)
            await anyio.sleep(self.misc_poll_interval)

    async def _monitor_system_profiler(self):
        if self.system_profiler_interval is None:
            return
        iface_map = await _gather_iface_map()
        if iface_map is None:
            return

        old_idents = []
        while True:
            data = await TBConnectivity.gather()
            assert data is not None

            idents = [it for i in data if (it := i.ident(iface_map)) is not None]
            if idents != old_idents:
                await self.info_sender.send(MacTBIdentifiers(idents=idents))
            old_idents = idents

            conns = [it for i in data if (it := i.conn()) is not None]
            await self.info_sender.send(MacTBConnections(conns=conns))

            await anyio.sleep(self.system_profiler_interval)

    async def _monitor_memory_usage(self):
        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )
        if self.memory_poll_rate is None:
            return
        while True:
            await self.info_sender.send(
                MemoryUsage.from_psutil(override_memory=override_memory)
            )
            await anyio.sleep(self.memory_poll_rate)

    async def _watch_system_info(self):
        if self.interface_watcher_interval is None:
            return
        old_nics = []
        while True:
            nics = get_network_interfaces()
            if nics != old_nics:
                old_nics = nics
                await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            await anyio.sleep(self.interface_watcher_interval)

    async def _monitor_linux_system(self):
        """Monitor Linux system metrics when macmon is not available."""
        if IS_DARWIN:
            return
        
        # Use the same interval as macmon
        interval = self.macmon_interval or 1.0
        
        while True:
            try:
                metrics = await LinuxSystemMetrics.gather()
                await self.info_sender.send(metrics)
            except Exception as e:
                logger.warning(f"Failed to gather Linux system metrics: {e}")
            
            await anyio.sleep(interval)

    async def _monitor_macmon(self, macmon_path: str):
        if self.macmon_interval is None:
            return
        # macmon pipe --interval [interval in ms]
        try:
            async with await open_process(
                [macmon_path, "pipe", "--interval", str(self.macmon_interval * 1000)]
            ) as p:
                if not p.stdout:
                    logger.critical("MacMon closed stdout")
                    return
                async for text in TextReceiveStream(
                    BufferedByteReceiveStream(p.stdout)
                ):
                    await self.info_sender.send(MacmonMetrics.from_raw_json(text))
        except CalledProcessError as e:
            stderr_msg = "no stderr"
            stderr_output = cast(bytes | str | None, e.stderr)
            if stderr_output is not None:
                stderr_msg = (
                    stderr_output.decode()
                    if isinstance(stderr_output, bytes)
                    else str(stderr_output)
                )
            logger.warning(
                f"MacMon failed with return code {e.returncode}: {stderr_msg}"
            )
