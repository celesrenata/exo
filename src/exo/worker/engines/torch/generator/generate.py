from typing import Callable, Generator

import torch

from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.torch import Model, TokenizerWrapper
from exo.worker.engines.torch.utils_torch import MAX_TOKENS, apply_chat_template
from exo.worker.runner.bootstrap import logger


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[torch.Tensor], torch.Tensor],
) -> int:
    """Warm up the PyTorch inference engine."""
    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content=content,
                )
            ],
        ),
    )

    tokens_generated = 0

    logger.info("Generating warmup tokens with PyTorch")

    # Tokenize the prompt
    input_ids = torch.tensor([tokenizer.encode(warmup_prompt)], dtype=torch.long)

    # Generate a few tokens for warmup
    with torch.no_grad():
        for _ in range(10):  # Generate 10 warmup tokens
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            next_token = sampler(logits)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            # Decode the new token
            token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            logger.info(f"Generated warmup token: {token_text}")
            tokens_generated += 1

            # Stop if we hit EOS
            if next_token.item() in tokenizer.eos_token_ids:
                break

    logger.info("Generated ALL warmup tokens")
    return tokens_generated


def torch_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[torch.Tensor], torch.Tensor],
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """Generate text using PyTorch model with streaming output."""
    logger.info(f"PyTorch generation task_params: {task}")

    prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    logger.info(f"Generated prompt: {prompt[:200]}...")

    # Tokenize the prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    max_tokens = task.max_tokens or MAX_TOKENS
    generated_tokens = 0

    logger.info(f"Starting generation with max_tokens={max_tokens}")

    with torch.no_grad():
        while generated_tokens < max_tokens:
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Sample next token
            next_token = sampler(logits)
            next_token_id = next_token.item()

            # Add token to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            generated_tokens += 1

            # Decode the new token
            try:
                # Decode just the new token for streaming
                token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)

                # Check for finish conditions
                finish_reason: FinishReason | None = None
                if next_token_id in tokenizer.eos_token_ids:
                    finish_reason = "stop"
                elif generated_tokens >= max_tokens:
                    finish_reason = "length"

                logger.debug(
                    f"Generated token {generated_tokens}: '{token_text}' (id={next_token_id})"
                )

                yield GenerationResponse(
                    text=token_text,
                    token=next_token_id,
                    finish_reason=finish_reason,
                )

                if finish_reason is not None:
                    logger.info(f"Generation finished: {finish_reason}")
                    break

            except Exception as e:
                logger.warning(f"Error decoding token {next_token_id}: {e}")
                # Skip this token and continue
                continue

    logger.info(f"Generation complete. Generated {generated_tokens} tokens.")


def torch_generate_simple(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Simple non-streaming generation for testing."""
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.tokenizer.pad_token_id,
            eos_token_id=tokenizer.tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = generated[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
