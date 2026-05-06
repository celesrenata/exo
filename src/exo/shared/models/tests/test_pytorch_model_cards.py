"""Unit tests for PyTorch-compatible model card TOML files.

Validates that all new PyTorch model cards load correctly, have the expected
field values, and coexist with existing MLX cards.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 5.1, 5.2
"""

from pathlib import Path

import pytest
from anyio import Path as AnyioPath

from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ModelCard, ModelTask

CARDS_DIR = Path(RESOURCES_DIR) / "inference_model_cards"

# All new PyTorch card filenames added by this spec
NEW_PYTORCH_CARD_FILES = [
    "Qwen--Qwen3.5-2B.toml",
    "meta-llama--Llama-3.2-1B-Instruct.toml",
    "Qwen--Qwen3.5-4B.toml",
    "Qwen--Qwen3.6-27B.toml",
    "Qwen--Qwen3.6-35B-A3B.toml",
    "zai-org--GLM-4.7-Flash.toml",
    "hugging-quants--Meta-Llama-3.3-70B-Instruct-GPTQ-INT4.toml",
]

# A sample of existing MLX cards to verify they still load
SAMPLE_MLX_CARD_FILES = [
    "mlx-community--Llama-3.2-1B-Instruct-4bit.toml",
    "mlx-community--Qwen3-0.6B-4bit.toml",
]


class TestNewCardFilesExist:
    """All expected new card files exist in resources/inference_model_cards/."""

    @pytest.mark.parametrize("filename", NEW_PYTORCH_CARD_FILES)
    def test_card_file_exists(self, filename: str) -> None:
        card_path = CARDS_DIR / filename
        assert card_path.exists(), f"Expected card file not found: {card_path}"


class TestNewCardsLoadWithoutValidationError:
    """Each new TOML card file loads without ValidationError."""

    @pytest.mark.parametrize("filename", NEW_PYTORCH_CARD_FILES)
    async def test_card_loads_successfully(self, filename: str) -> None:
        card_path = AnyioPath(str(CARDS_DIR / filename))
        card = await ModelCard.load_from_path(card_path)
        assert card.model_id != ""
        assert card.n_layers > 0
        assert card.hidden_size > 0
        assert card.storage_size.in_bytes > 0


class TestSmallCardProperties:
    """Small cards have quantization == "" and TextGeneration in tasks.

    Validates: Requirements 1.1, 1.2, 1.4, 1.5
    """

    SMALL_CARDS = [
        "Qwen--Qwen3.5-2B.toml",
        "meta-llama--Llama-3.2-1B-Instruct.toml",
    ]

    @pytest.mark.parametrize("filename", SMALL_CARDS)
    async def test_small_card_quantization_empty(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert card.quantization == "", (
            f"{filename}: expected empty quantization, got {card.quantization!r}"
        )

    @pytest.mark.parametrize("filename", SMALL_CARDS)
    async def test_small_card_has_text_generation(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert ModelTask.TextGeneration in card.tasks, (
            f"{filename}: expected TextGeneration in tasks, got {card.tasks}"
        )

    @pytest.mark.parametrize("filename", SMALL_CARDS)
    async def test_small_card_storage_under_4gib(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        four_gib = 4 * 1024**3
        assert card.storage_size.in_bytes <= four_gib, (
            f"{filename}: storage {card.storage_size.in_bytes} exceeds 4 GiB"
        )


class TestGPTQCardProperties:
    """GPTQ cards have quantization set correctly.

    Validates: Requirements 2.3, 3.3
    """

    async def test_gptq_llama_quantization(self) -> None:
        filename = "hugging-quants--Meta-Llama-3.3-70B-Instruct-GPTQ-INT4.toml"
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert card.quantization == "GPTQ-Int4", (
            f"Expected quantization='GPTQ-Int4', got {card.quantization!r}"
        )

    async def test_gptq_card_has_text_generation(self) -> None:
        filename = "hugging-quants--Meta-Llama-3.3-70B-Instruct-GPTQ-INT4.toml"
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert ModelTask.TextGeneration in card.tasks


class TestSupportsTensorByArchitecture:
    """supports_tensor is correct per model architecture.

    Llama models have supports_tensor=true, Qwen/GLM models have supports_tensor=false.

    Validates: Requirements 1.3, 3.3
    """

    LLAMA_CARDS = [
        "meta-llama--Llama-3.2-1B-Instruct.toml",
        "hugging-quants--Meta-Llama-3.3-70B-Instruct-GPTQ-INT4.toml",
    ]

    QWEN_AND_GLM_CARDS = [
        "Qwen--Qwen3.6-27B.toml",
        "Qwen--Qwen3.6-35B-A3B.toml",
        "zai-org--GLM-4.7-Flash.toml",
    ]

    QWEN_TENSOR_CARDS = [
        "Qwen--Qwen3.5-2B.toml",
        "Qwen--Qwen3.5-4B.toml",
    ]

    @pytest.mark.parametrize("filename", LLAMA_CARDS)
    async def test_llama_supports_tensor_true(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert card.supports_tensor is True, (
            f"{filename}: expected supports_tensor=true for Llama"
        )

    @pytest.mark.parametrize("filename", QWEN_AND_GLM_CARDS)
    async def test_qwen_glm_supports_tensor_false(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert card.supports_tensor is False, (
            f"{filename}: expected supports_tensor=false for Qwen/GLM"
        )

    @pytest.mark.parametrize("filename", QWEN_TENSOR_CARDS)
    async def test_qwen35_supports_tensor_true(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        assert card.supports_tensor is True, (
            f"{filename}: expected supports_tensor=true for Qwen3.5 (tensor parallelism enabled)"
        )


class TestMLXCardsStillLoad:
    """Existing MLX cards still load correctly alongside new PyTorch cards.

    Validates: Requirements 4.1, 4.2
    """

    @pytest.mark.parametrize("filename", SAMPLE_MLX_CARD_FILES)
    async def test_mlx_card_loads(self, filename: str) -> None:
        card_path = CARDS_DIR / filename
        if not card_path.exists():
            pytest.skip(f"MLX card {filename} not present in this checkout")
        card = await ModelCard.load_from_path(AnyioPath(str(card_path)))
        assert card.model_id.startswith("mlx-community/")
        assert ModelTask.TextGeneration in card.tasks


class TestCardFieldCompleteness:
    """All required fields are present in each new card.

    Validates: Requirements 5.1, 5.2
    """

    @pytest.mark.parametrize("filename", NEW_PYTORCH_CARD_FILES)
    async def test_all_required_fields_present(self, filename: str) -> None:
        card = await ModelCard.load_from_path(AnyioPath(str(CARDS_DIR / filename)))
        # These fields are required by the ModelCard schema
        assert card.model_id != ""
        assert card.storage_size.in_bytes > 0
        assert card.n_layers > 0
        assert card.hidden_size > 0
        assert isinstance(card.supports_tensor, bool)
        assert len(card.tasks) > 0
        # These have defaults but should be explicitly set in PyTorch cards
        assert card.family != "", f"{filename}: family should be set"
        assert card.base_model != "", f"{filename}: base_model should be set"


# ---------------------------------------------------------------------------
# Property-based test: ModelCard TOML round-trip
# ---------------------------------------------------------------------------

import asyncio
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.shared.types.memory import Memory


def _model_id_strategy() -> st.SearchStrategy[str]:
    """Generate valid model IDs in org/model format."""
    segment = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=48, max_codepoint=122),
        min_size=1,
        max_size=20,
    )
    return st.tuples(segment, segment).map(lambda t: f"{t[0]}/{t[1]}")


def _simple_text() -> st.SearchStrategy[str]:
    """Generate simple ASCII text safe for TOML round-tripping."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), min_codepoint=32, max_codepoint=122),
        min_size=0,
        max_size=30,
    )


@st.composite
def model_card_strategy(draw: st.DrawFn) -> ModelCard:
    """Generate a random valid ModelCard instance."""
    model_id = draw(_model_id_strategy())
    storage_size = Memory(in_bytes=draw(st.integers(min_value=1, max_value=10**12)))
    n_layers = draw(st.integers(min_value=1, max_value=200))
    hidden_size = draw(st.integers(min_value=1, max_value=65536))
    supports_tensor = draw(st.booleans())
    tasks = draw(st.lists(st.sampled_from(list(ModelTask)), min_size=1, max_size=3, unique=True))
    family = draw(_simple_text())
    quantization = draw(_simple_text())
    base_model = draw(_simple_text())
    capabilities = draw(st.lists(_simple_text(), min_size=0, max_size=5))
    uses_cfg = draw(st.booleans())

    return ModelCard(
        model_id=model_id,
        storage_size=storage_size,
        n_layers=n_layers,
        hidden_size=hidden_size,
        supports_tensor=supports_tensor,
        tasks=tasks,
        family=family,
        quantization=quantization,
        base_model=base_model,
        capabilities=capabilities,
        uses_cfg=uses_cfg,
    )


class TestModelCardTomlRoundTrip:
    """Property 1: ModelCard TOML round-trip.

    For any valid ModelCard instance, serializing via save() to a TOML file
    and re-parsing via load_from_path() produces an equivalent ModelCard.

    **Validates: Requirements 1.1, 5.1, 5.3, 5.4**
    """

    @given(card=model_card_strategy())
    @settings(max_examples=100)
    def test_round_trip_preserves_all_fields(self, card: ModelCard) -> None:
        """Serialize a ModelCard to TOML and re-parse; all fields must match."""

        async def _round_trip() -> None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                toml_path = AnyioPath(tmp_dir) / "card.toml"
                await card.save(toml_path)
                loaded = await ModelCard.load_from_path(toml_path)

                assert loaded.model_id == card.model_id
                assert loaded.storage_size.in_bytes == card.storage_size.in_bytes
                assert loaded.n_layers == card.n_layers
                assert loaded.hidden_size == card.hidden_size
                assert loaded.supports_tensor == card.supports_tensor
                assert loaded.tasks == card.tasks
                assert loaded.family == card.family
                assert loaded.quantization == card.quantization
                assert loaded.base_model == card.base_model
                assert loaded.capabilities == card.capabilities
                assert loaded.uses_cfg == card.uses_cfg
                assert loaded.components == card.components

        asyncio.run(_round_trip())
