"""Architecture documentation generator with Mermaid diagrams.

Generates docs/architecture.md with component sections, interaction diagrams,
and event sourcing message flow details by combining hardcoded component
knowledge with AST analysis of topic definitions.
"""

import ast
from pathlib import Path
from typing import Final

from loguru import logger

from exo.docgen.model_client import chat_completion
from exo.docgen.models import PlannedWrite

# Path to the topics definition file relative to project root
_TOPICS_FILE: Final[str] = "src/exo/routing/topics.py"

# The 5 core components
_COMPONENTS: Final[list[str]] = ["Router", "Worker", "Master", "Election", "API"]

# Hardcoded component-topic relationships
_COMPONENT_PUBLISHES: Final[dict[str, list[str]]] = {
    "Router": ["CONNECTION_MESSAGES"],
    "Worker": ["LOCAL_EVENTS"],
    "Master": ["GLOBAL_EVENTS", "COMMANDS"],
    "Election": ["ELECTION_MESSAGES"],
    "API": ["COMMANDS"],
}

_COMPONENT_SUBSCRIBES: Final[dict[str, list[str]]] = {
    "Router": [
        "GLOBAL_EVENTS",
        "LOCAL_EVENTS",
        "COMMANDS",
        "ELECTION_MESSAGES",
        "CONNECTION_MESSAGES",
    ],
    "Worker": ["GLOBAL_EVENTS", "COMMANDS", "DOWNLOAD_COMMANDS"],
    "Master": ["LOCAL_EVENTS", "COMMANDS", "ELECTION_MESSAGES"],
    "Election": ["ELECTION_MESSAGES", "CONNECTION_MESSAGES"],
    "API": ["GLOBAL_EVENTS"],
}


class _TopicInfo:
    """Extracted topic information from AST analysis."""

    __slots__ = ("name", "topic_string", "publish_policy", "message_type")

    def __init__(
        self,
        name: str,
        topic_string: str,
        publish_policy: str,
        message_type: str,
    ) -> None:
        self.name = name
        self.topic_string = topic_string
        self.publish_policy = publish_policy
        self.message_type = message_type


def _parse_topics_from_ast(project_root: Path) -> list[_TopicInfo]:
    """Parse topic definitions from src/exo/routing/topics.py using AST.

    Extracts topic variable assignments that call TypedTopic(...) and pulls
    out the topic string, publish policy, and message type arguments.
    """
    topics_path = project_root / _TOPICS_FILE
    if not topics_path.exists():
        logger.warning("Topics file not found at {}", topics_path)
        return []

    try:
        source = topics_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(topics_path))
    except (OSError, SyntaxError) as error:
        logger.warning("Failed to parse topics file {}: {}", topics_path, error)
        return []

    topics: list[_TopicInfo] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue

        # Check if the value is a call to TypedTopic(...)
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        if not isinstance(func, ast.Name) or func.id != "TypedTopic":
            continue

        # Extract positional arguments: topic_string, publish_policy, message_type
        if len(value.args) < 3:
            continue

        topic_string_node = value.args[0]
        policy_node = value.args[1]
        message_type_node = value.args[2]

        # Extract topic string
        topic_string = ""
        if isinstance(topic_string_node, ast.Constant) and isinstance(
            topic_string_node.value, str
        ):
            topic_string = topic_string_node.value

        # Extract publish policy (PublishPolicy.Always, etc.)
        publish_policy = ""
        if isinstance(policy_node, ast.Attribute):
            publish_policy = policy_node.attr

        # Extract message type name
        message_type = ""
        if isinstance(message_type_node, ast.Name):
            message_type = message_type_node.id

        topics.append(
            _TopicInfo(
                name=target.id,
                topic_string=topic_string,
                publish_policy=publish_policy,
                message_type=message_type,
            )
        )

    return topics


def _build_component_prompt() -> str:
    """Build a prompt asking the model to generate component role summaries."""
    component_descriptions = "\n".join(
        f"- {name}: publishes to {', '.join(_COMPONENT_PUBLISHES[name])}, "
        f"subscribes to {', '.join(_COMPONENT_SUBSCRIBES[name])}"
        for name in _COMPONENTS
    )

    return (
        "Generate a one-paragraph role summary (3-4 sentences) for each of the "
        "following components in a distributed AI inference system called 'exo'. "
        "The system uses event sourcing with pub/sub messaging for coordination.\n\n"
        "Components and their topic relationships:\n"
        f"{component_descriptions}\n\n"
        "Context:\n"
        "- Router: libp2p-based pub/sub messaging layer using Rust bindings\n"
        "- Worker: Handles inference tasks, downloads models, manages runner processes\n"
        "- Master: Coordinates cluster state, places model instances across nodes\n"
        "- Election: Bully algorithm for master election among peers\n"
        "- API: FastAPI server for OpenAI-compatible chat completions\n\n"
        "Format your response as:\n"
        "## Router\n<paragraph>\n\n## Worker\n<paragraph>\n\n"
        "## Master\n<paragraph>\n\n## Election\n<paragraph>\n\n## API\n<paragraph>"
    )


def _generate_mermaid_diagram(topics: list[_TopicInfo]) -> str:
    """Generate a Mermaid flowchart showing component interactions via topics."""
    lines: list[str] = ["```mermaid", "flowchart LR"]

    # Define component nodes
    for component in _COMPONENTS:
        node_id = component.lower()
        lines.append(f"    {node_id}[{component}]")

    lines.append("")

    # Add edges for publish relationships
    topic_names = {t.name for t in topics}
    for component in _COMPONENTS:
        source_id = component.lower()
        for topic_name in _COMPONENT_PUBLISHES.get(component, []):
            if topic_name not in topic_names:
                continue
            # Find subscribers for this topic
            for subscriber in _COMPONENTS:
                if subscriber == component:
                    continue
                if topic_name in _COMPONENT_SUBSCRIBES.get(subscriber, []):
                    target_id = subscriber.lower()
                    label = topic_name.replace("_", " ").title()
                    lines.append(f"    {source_id} -->|{label}| {target_id}")

    lines.append("```")
    return "\n".join(lines)


def _generate_message_flow_table(topics: list[_TopicInfo]) -> str:
    """Generate a markdown table describing the event sourcing message flow."""
    lines: list[str] = [
        "| Topic | Publishing Component | Subscribing Component(s) "
        "| Message Type | Publish Policy |",
        "|-------|---------------------|--------------------------|"
        "--------------|----------------|",
    ]

    for topic in topics:
        # Find publishers
        publishers: list[str] = []
        for component in _COMPONENTS:
            if topic.name in _COMPONENT_PUBLISHES.get(component, []):
                publishers.append(component)

        # Find subscribers
        subscribers: list[str] = []
        for component in _COMPONENTS:
            if topic.name in _COMPONENT_SUBSCRIBES.get(component, []):
                subscribers.append(component)

        publisher_str = ", ".join(publishers) if publishers else "—"
        subscriber_str = ", ".join(subscribers) if subscribers else "—"

        lines.append(
            f"| {topic.name} | {publisher_str} | {subscriber_str} "
            f"| `{topic.message_type}` | {topic.publish_policy} |"
        )

    return "\n".join(lines)


def _generate_component_sections(
    model_response: str | None,
    topics: list[_TopicInfo],
) -> str:
    """Generate the component sections with role summaries and topic lists."""
    sections: list[str] = []

    # Parse model response into per-component summaries
    component_summaries: dict[str, str] = {}
    if model_response:
        current_component: str | None = None
        current_lines: list[str] = []
        for line in model_response.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_component and current_lines:
                    component_summaries[current_component] = "\n".join(
                        current_lines
                    ).strip()
                current_component = stripped.removeprefix("## ").strip()
                current_lines = []
            elif current_component:
                current_lines.append(line)
        if current_component and current_lines:
            component_summaries[current_component] = "\n".join(
                current_lines
            ).strip()

    # Fallback summaries if model didn't respond
    fallback_summaries: dict[str, str] = {
        "Router": (
            "The Router component provides the libp2p-based pub/sub messaging "
            "layer using Rust bindings (exo_pyo3_bindings). It handles message "
            "serialization, deserialization, and routing between all components "
            "in the cluster. It publishes connection state changes and routes "
            "all other topic messages between local and remote peers."
        ),
        "Worker": (
            "The Worker component handles inference tasks, downloads models, "
            "and manages runner processes. It receives commands and global events "
            "from the Master, executes inference workloads on available hardware, "
            "and reports local events back to the Master for state indexing."
        ),
        "Master": (
            "The Master component coordinates cluster state and places model "
            "instances across nodes. It receives local events from Workers, "
            "indexes them into the global event log, and broadcasts indexed "
            "events to all peers. It also issues commands for task assignment "
            "and model placement."
        ),
        "Election": (
            "The Election component implements a bully algorithm for master "
            "election among peers. It monitors connection state changes to "
            "detect when a new election is needed and exchanges election "
            "protocol messages with other nodes to determine the cluster leader."
        ),
        "API": (
            "The API component provides a FastAPI server implementing "
            "OpenAI-compatible chat completion endpoints. It accepts inference "
            "requests from external clients, translates them into commands for "
            "the Master, and subscribes to global events to track request "
            "progress and stream responses."
        ),
    }

    topic_names_set = {t.name for t in topics}

    for component in _COMPONENTS:
        summary = component_summaries.get(
            component, fallback_summaries.get(component, "")
        )
        publishes = [
            t
            for t in _COMPONENT_PUBLISHES.get(component, [])
            if t in topic_names_set
        ]
        subscribes = [
            t
            for t in _COMPONENT_SUBSCRIBES.get(component, [])
            if t in topic_names_set
        ]

        section_lines: list[str] = [
            f"## {component}",
            "",
            summary,
            "",
        ]

        if publishes:
            section_lines.append("**Publishes to:**")
            for topic_name in publishes:
                section_lines.append(f"- `{topic_name}`")
            section_lines.append("")

        if subscribes:
            section_lines.append("**Subscribes to:**")
            for topic_name in subscribes:
                section_lines.append(f"- `{topic_name}`")
            section_lines.append("")

        sections.append("\n".join(section_lines))

    return "\n".join(sections)


async def generate_architecture_documentation(
    output_directory: Path,
    project_root: Path | None = None,
) -> PlannedWrite | None:
    """Generate architecture documentation with Mermaid diagrams.

    Combines hardcoded component knowledge with AST analysis of topic
    definitions to produce a complete docs/architecture.md file.

    Args:
        output_directory: Directory where architecture.md will be written.
        project_root: Root of the project for locating topics.py.
            Defaults to current working directory.

    Returns:
        A PlannedWrite with the generated content, or None if generation fails.
    """
    if project_root is None:
        project_root = Path.cwd()

    # Parse topics from AST
    topics = _parse_topics_from_ast(project_root)
    if not topics:
        logger.warning(
            "No topics found in {}; using hardcoded topic information",
            project_root / _TOPICS_FILE,
        )
        # Provide hardcoded fallback topic info
        topics = [
            _TopicInfo(
                "GLOBAL_EVENTS", "global_events", "Always", "GlobalForwarderEvent"
            ),
            _TopicInfo(
                "LOCAL_EVENTS", "local_events", "Always", "LocalForwarderEvent"
            ),
            _TopicInfo("COMMANDS", "commands", "Always", "ForwarderCommand"),
            _TopicInfo(
                "ELECTION_MESSAGES",
                "election_messages",
                "Always",
                "ElectionMessage",
            ),
            _TopicInfo(
                "CONNECTION_MESSAGES",
                "connection_messages",
                "Never",
                "ConnectionMessage",
            ),
            _TopicInfo(
                "DOWNLOAD_COMMANDS",
                "download_commands",
                "Always",
                "ForwarderDownloadCommand",
            ),
        ]

    # Request component summaries from the model
    system_prompt = (
        "You are a technical documentation writer for a distributed systems "
        "project. Write clear, concise descriptions of software components. "
        "Use present tense and active voice."
    )
    user_prompt = _build_component_prompt()

    model_response = await chat_completion("condense", system_prompt, user_prompt)
    if model_response is None:
        logger.warning(
            "Model failed to generate component summaries; "
            "using fallback descriptions"
        )

    # Build the full markdown document
    mermaid_diagram = _generate_mermaid_diagram(topics)
    message_flow_table = _generate_message_flow_table(topics)
    component_sections = _generate_component_sections(model_response, topics)

    content_parts: list[str] = [
        "# Architecture Overview",
        "",
        "This document describes the high-level architecture of the exo "
        "distributed AI inference system, including component roles, pub/sub "
        "topic interactions, and event sourcing message flow.",
        "",
        "## Component Interaction Diagram",
        "",
        mermaid_diagram,
        "",
        component_sections,
        "## Event Sourcing Message Flow",
        "",
        "The system uses event sourcing for state management. Components "
        "communicate via typed pub/sub topics. The Master indexes local events "
        "into the global event log and broadcasts them to all peers. Workers "
        "apply indexed events to maintain consistent state.",
        "",
        message_flow_table,
        "",
    ]

    content = "\n".join(content_parts)

    # Determine action based on whether the file already exists
    destination = output_directory / "architecture.md"
    existing_content: str | None = None
    action: str = "create"

    if destination.exists():
        existing_content = destination.read_text(encoding="utf-8")
        action = "update"

    return PlannedWrite(
        destination=destination,
        content=content,
        action=action,
        existing_content=existing_content,
    )
