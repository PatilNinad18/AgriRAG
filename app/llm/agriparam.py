"""
app/llm/agriparam.py
---------------------
Configurable causal language model integration.

Two prompt modes:
  1. RAG mode   : PDF context found → grounded answer with citations
  2. Fallback   : No PDF context   → model answers from its pretrained knowledge
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

# Used when PDF context IS available
RAG_PROMPT_TEMPLATE = """\
You are AgriAssist, a highly knowledgeable agricultural advisor specialising in Indian farming.

You have been provided with relevant excerpts from agricultural documents below.
Use this information as the primary source for your answer.
If the context does not fully cover the question, supplement it with your own agricultural expertise.

Provide a structured, detailed answer covering:
1. Overview / Introduction
2. Step-by-step method or key recommendations
3. Specific details (varieties, doses, seasons, soil type, irrigation, etc.)
4. Common problems and solutions (if relevant)

Context from knowledge base:
\"\"\"
{context}
\"\"\"

Farmer's Question: {question}

Detailed Expert Answer:
"""

# Used when NO PDF context is found (pure model knowledge)
NO_CONTEXT_PROMPT_TEMPLATE = """\
You are AgriAssist, a highly knowledgeable agricultural advisor specialising in Indian farming.

The farmer has asked a question that is not covered in the current knowledge base documents.
Use your expert agricultural knowledge to provide a thorough and helpful answer.

Provide a structured, detailed answer covering:
1. Overview / Introduction
2. Step-by-step method or key recommendations
3. Specific details (varieties, doses, seasons, soil type, irrigation, etc.)
4. Common problems and solutions (if relevant)

Note at the end: "Note: This answer is based on general agricultural knowledge. \
For region-specific advice, consult your local Krishi Vigyan Kendra (KVK)."

Farmer's Question: {question}

Detailed Expert Answer:
"""


def _load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load the tokenizer bundled with the configured model."""
    logger.info("Loading tokenizer", model=model_path)
    return AutoTokenizer.from_pretrained(model_path)


class AgriParamService:
    """Manages the configured language model lifecycle and generation."""

    def __init__(self) -> None:
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        """Load the configured model using the selected device and precision."""
        # Use local model path if provided, otherwise use HuggingFace model name
        model_source = settings.agriparam_model_path or settings.agriparam_model_name

        logger.info(
            "Loading language model",
            model_source=model_source,
            device=settings.agriparam_device,
        )

        self._tokenizer = _load_tokenizer(model_source)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float16 if settings.agriparam_device == "cuda" else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map=settings.agriparam_device,
        )
        self._model.eval()

        if settings.agriparam_device == "cuda" and torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                "GPU VRAM usage after model load",
                used_gb=round(vram_used, 2),
                total_gb=round(vram_total, 2),
            )

        logger.info(
            "Language model loaded",
            model_source=model_source,
            device=settings.agriparam_device,
        )

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            raise RuntimeError("Language model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Language model tokenizer not loaded. Call load() first.")
        return self._tokenizer

    def generate(self, question: str, context: str = "") -> str:
        """
        Generate a detailed agriculture answer.

        If context is provided (from PDF retrieval) → RAG mode.
        If context is empty (no matching documents)  → fallback to model knowledge.
        """
        using_context = bool(context.strip())

        if using_context:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )
            logger.info("Generating answer in RAG mode (context from PDF)")
        else:
            prompt = NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)
            logger.info("Generating answer in fallback mode (no PDF context found)")

        add_special_tokens = True
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            add_special_tokens = False

        # Tokenise — use a larger max_length to avoid cutting off the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,   # Larger window: 1536 prompt + 400 new = ~1936 total
            padding=True,
            add_special_tokens=add_special_tokens,
        ).to(settings.agriparam_device)

        prompt_token_count = inputs["input_ids"].shape[-1]
        logger.debug(
            "Generating answer",
            prompt_tokens=prompt_token_count,
            max_new_tokens=settings.agriparam_max_new_tokens,
            mode="rag" if using_context else "fallback",
        )

        gen_config = GenerationConfig(
            max_new_tokens=settings.agriparam_max_new_tokens,
            temperature=settings.agriparam_temperature,
            top_p=settings.agriparam_top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        # Decode only newly generated tokens
        generated_ids = output_ids[0][prompt_token_count:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = answer.strip()

        # If answer still seems cut off (ends mid-word), append a note
        if answer and not answer[-1] in ".!?":
            answer = answer.rstrip(",;: ") + "."

        logger.debug(
            "Answer generated",
            length=len(answer),
            mode="rag" if using_context else "fallback",
        )
        return answer

    def health_check(self) -> bool:
        try:
            _ = self.generate("What is a Kharif crop?", context="")
            return True
        except Exception:
            return False


# Module-level singleton
agriparam_service = AgriParamService()
