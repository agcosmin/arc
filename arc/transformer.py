import dataclasses

import torch.nn


@dataclasses.dataclass
class ARCEncoderConfig:
    num_token_ids: int
    num_positions: int = 30 * 31 * 4 + 8  # 2 example of max size w/ run 1
    embedding_dim: int = 512
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-05
    activation: str = "relu"
    num_layers: int = 2
    padding_index: int = 0
    max_embedding_norm: float | None = None
    num_problem_types: int = -1
    tokenizer_max_run_length: int = 30


class ARCEncoder(torch.nn.Module):
    def __init__(self, config: ARCEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=config.num_token_ids,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_index,
            max_norm=config.max_embedding_norm,
        )
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=config.num_positions,
            embedding_dim=config.embedding_dim,
            max_norm=config.max_embedding_norm,
        )
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                bias=True,
            ),
            num_layers=config.num_layers,
        )
        self.lm_head = torch.nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=config.num_token_ids,
            bias=True,
        )
        self.problem_type_head = None
        self.add_problem_type_pred_head(config.num_problem_types)

    def add_problem_type_pred_head(self, num_problem_types: int) -> None:
        if num_problem_types > 0:
            self.problem_type_head = torch.nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=num_problem_types,
                bias=True,
            )
            self.config.num_problem_types = num_problem_types

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        return_lm_logits: bool = True,
        return_problem_type_logits: bool = False,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        if return_problem_type_logits and not self.problem_type_head:
            raise ValueError("Model does not have problem type prediction head")

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(
            torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        )
        embeddings = token_embeddings + position_embeddings
        embeddings = self.encoder(
            embeddings,
            mask=torch.nn.Transformer.generate_square_subsequent_mask(
                input_ids.shape[1], device=input_ids.device
            ),
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        outputs = {}
        if return_lm_logits:
            outputs["lm_logits"] = self.lm_head(embeddings)

        if return_problem_type_logits:
            outputs["problem_type_logits"] = self.problem_type_head(embeddings)

        if return_embeddings:
            outputs["embeddings"] = embeddings

        return outputs
