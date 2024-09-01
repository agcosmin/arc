import typing

import torch

import arc.transformer
import arc.tokenizer
import arc.dataset


@torch.no_grad
def generate_solution(
    context: torch.Tensor,
    model: arc.transformer.ARCEncoder,
    tokenizer: arc.tokenizer.ARCTokenizer,
    shape: typing.Optional[tuple[int, int]] = None,
    max_shape: tuple[int, int] = (30, 30),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if shape is not None and (shape[0] <= 0 or shape[1] <= 0):
        raise ValueError("Shape values must be > 1.")
    if max_shape[0] <= 0 or max_shape[1] <= 0:
        raise ValueError("Shape values must be > 1.")
    model.eval()
    solution = torch.tensor([], dtype=context.dtype, device=device)
    context = context.to(device)
    if shape is None:
        shape = max_shape
        free_form = True
    else:
        free_form = False
    num_rows = num_cols = 0
    while num_rows < shape[0]:
        num_cols = 0
        while num_cols < shape[1]:
            input_ids = torch.hstack((context, solution)).unsqueeze(0)
            logits = model(
                input_ids=input_ids,
                padding_mask=torch.zeros(input_ids.shape, device=device),
            )["lm_logits"]
            top_tokens = torch.topk(
                logits[:, -1:, :], len(tokenizer.special_tokens) + 1
            ).indices.flatten()
            for token in top_tokens:
                if token.item() not in tokenizer.special_tokens.values():
                    value, run = tokenizer.decode_token(token)
                    if num_cols + run > shape[1]:
                        run = shape[1] - num_cols
                        token = torch.tensor(
                            tokenizer.get_token_id(value, run), device=device
                        )
                    num_cols += run
                    solution = torch.hstack((solution, token))
                    break
                elif free_form:
                    if (
                        token.item() == tokenizer.special_tokens["<rhs>"]
                        and num_cols > 0
                    ):
                        num_cols = shape[1]
                        break
                    elif token.item() == tokenizer.special_tokens["<eoo>"] and (
                        num_cols > 0 or num_rows > 0
                    ):
                        num_rows = shape[0]
                        num_cols = shape[1]
                        break
            if (
                num_cols == shape[1]
                and solution[-1].item() != tokenizer.special_tokens["<rhs>"]
            ):
                solution = torch.hstack(
                    (
                        solution,
                        torch.tensor(
                            tokenizer.special_tokens["<rhs>"], device=device
                        ),
                    )
                )
        num_rows += 1
    solution = torch.hstack(
        (
            torch.tensor(tokenizer.special_tokens["<boo>"], device=device),
            solution,
            torch.tensor(tokenizer.special_tokens["<eoo>"], device=device),
        )
    )
    solution = solution.cpu()
    return solution


def main():
    tokenizer_max_run_length = 30
    tokenizer = arc.tokenizer.ARCTokenizer(
        max_run_length=tokenizer_max_run_length
    )
    dataset = arc.dataset.ARCDataset(
        [
            (
                "./data/arc-agi_training_challenges.json",
                "./data/arc-agi_training_solutions.json",
            )
        ],
        train=False,
        tokenizer=tokenizer,
        augment=arc.transform.Augment(
            value_permutation=True,
            fliplr=True,
            flipud=True,
            rotate=True,
            num_example_samples=20,
            limit_run=False,
        ),
        return_raw=True,
    )
    config = arc.transformer.ARCEncoderConfig(
        num_token_ids=tokenizer.num_token_ids,
        embedding_dim=512,
        num_heads=8,
        dim_feedforward=2048,
        num_layers=4,
        tokenizer_max_run_length=tokenizer_max_run_length,
    )
    model = arc.transformer.ARCEncoder(config)
    checkpoint_path = "./experiments/2024_08_30_23_26_55/checkpoints/last.ckpt"
    model.load_state_dict(
        {
            k.replace("encoder.", "", 1): v
            for k, v in torch.load(checkpoint_path, weights_only=True)[
                "state_dict"
            ].items()
        }
    )
    device = torch.device("cuda")
    model = model.to(device)
    num_samples = 10
    for problem_i in range(len(dataset)):
        solutions = []
        for sample in range(num_samples):
            tests_solutions = []
            for test in dataset[problem_i]:
                solution = generate_solution(
                    context=test["tokenized_sequence"],
                    model=model,
                    tokenizer=tokenizer,
                    shape=None,
                    device=device,
                )
                tests_solutions.append(tokenizer.decode(solution))
            solutions.append(tests_solutions)


if __name__ == "__main__":
    main()
