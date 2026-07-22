from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "bert-base-multilingual-cased"

def get_device() -> torch.device:
    """
    Choose the best available device. (CPU, CUDA, MPS)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """
    Load Hugging Face tokenizer and transformer model.

    Returns:
        tokenizer: Converts text into token IDs.
        model: Converts token IDs into contextual token embeddings.
        device: CPU/GPU/MPS device used by the model.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return tokenizer, model, device

def mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert token-level embeddings into one sentence-level embedding.

    BERT returns an embedding for each token. Average the token embeddings while ignoring padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * mask, dim=1)
    token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    sentence_embeddings = summed_embeddings / token_counts
    return sentence_embeddings

def cls_pool(token_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Use CLS token embedding as the sentence-level embedding. (In BERT-style models, the first token is a special CLS token.)
    """
    return token_embeddings[:, 0, :]

def embed_sentences(
    sentences: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    normalize: bool = True,
    pooling_method: str = "mean",
) -> np.ndarray:
    """
    Convert a list of sentences into a matrix of sentence embeddings.

    Returns:
        A numpy array of shape [num_sentences, embedding_dimension].
    """
    sentences = list(sentences)
    all_embeddings: List[np.ndarray] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        if pooling_method == "mean":
            sentence_embeddings = mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
        elif pooling_method == "cls":
            sentence_embeddings = cls_pool(outputs.last_hidden_state)

        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        all_embeddings.append(sentence_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors. (Dot product of normalized embeddings.)
    """
    embedding_a = np.asarray(embedding_a)
    embedding_b = np.asarray(embedding_b)

    denominator = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)

    if denominator == 0:
        raise ValueError("Zero vector.")

    return float(np.dot(embedding_a, embedding_b) / denominator)

def create_embedding_lookup(
    sentences: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    pooling_method: str = "mean"
) -> Dict[str, np.ndarray]:
    """
    Create a dictionary mapping each unique sentence to its embedding. Embed each unique sentence once.
    """
    unique_sentences = list(dict.fromkeys(sentences))

    embeddings = embed_sentences(
        unique_sentences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        pooling_method=pooling_method,
    )

    return {
        sentence: embedding
        for sentence, embedding in zip(unique_sentences, embeddings)
    }