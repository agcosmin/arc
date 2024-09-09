import argparse
import json
import typing

import torch
import tqdm

ON_KAGGLE = False
if ON_KAGGLE:
    pass
else:
    import arc.transformer
    import arc.tokenizer
    import arc.dataset

KAGGLE_ARGS = ["arc-agi_test-challenges.json", "submission.json", "model"]


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


def solve_problem(
    examples: list[dict[str, torch.Tensor]],
    test: dict[str, torch.Tensor],
    tokenizer: arc.tokenizer.ARCTokenizer,
    model: arc.transformer.ARCEncoder,
    num_samples: int = 500,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    augment = arc.transform.Augment(
        value_permutation=True,
        fliplr=True,
        flipud=True,
        rotate=True,
        num_example_samples=10,
        limit_run=False,
    )
    max_length = 30 * 31 * 3 + 7  # 1 example + 1 input of max size w/ run 1
    shape_ratios = torch.vstack(
        [
            torch.tensor(e["output"].shape) / torch.tensor(e["input"].shape)
            for e in examples
        ]
    )
    same_ratio = torch.all(shape_ratios == shape_ratios[0])
    if same_ratio:
        output_shape = (
            (torch.tensor(test["input"].shape) * shape_ratios[0])
            .to(torch.int)
            .tolist()
        )
    else:
        output_shape = None

    solutions = []
    for _ in range(num_samples):
        context, *_ = arc.transform.sample_sequence(
            examples=examples,
            augment=augment,
            tokenizer=tokenizer,
            test=test,
            max_length=max_length,
        )
        solution = tokenizer.decode(
            generate_solution(
                context=context,
                model=model,
                tokenizer=tokenizer,
                shape=output_shape,
                device=device,
            )
        )[0]
        solutions.append(solution)

    if output_shape is not None:
        solution = torch.stack(solutions, dim=-1)
    else:
        shapes = torch.vstack([torch.tensor(t.shape) for t in solutions])
        output_shape = shapes.median(dim=0).values
        ends = torch.min(shapes, output_shape).tolist()
        solution = torch.zeros(
            output_shape.tolist() + [len(solutions)], dtype=solution[0].dtype
        )
        for i, img in enumerate(solutions):
            end_row, end_col = ends[i]
            solution[0:end_row, 0:end_col, i] = img[0:end_row, 0:end_col]

    solution = torch.mode(solution, dim=-1).values
    return solution


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("challanges", type=str, help="Path to challanges.")
    parser.add_argument("output", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--value-permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fliplr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--flipud", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--rotate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--limit-run", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-example-samples", type=int, default=10)
    parser.add_argument(
        "--solutions", type=str, default=None, help="Path to solutions."
    )
    return parser


def main():
    args = create_argparser().parse_args(KAGGLE_ARGS if ON_KAGGLE else None)
    dataset = arc.dataset.ARCDataset([(args.challanges, args.solutions)])
    checkpoint = torch.load(args.model)
    encoder_config = checkpoint["hyper_parameters"]["config"]
    tokenizer = arc.tokenizer.ARCTokenizer(
        max_run_length=encoder_config.tokenizer_max_run_length
    )
    model = arc.transformer.ARCEncoder(encoder_config)
    model.eval()
    model.load_state_dict(
        {
            k.replace("encoder.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k != "lm_loss.weight"
        }
    )
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    augment = arc.transform.Augment(
        value_permutation=args.value_permutation,
        fliplr=args.fliplr,
        flipud=args.flipud,
        rotate=args.rotate,
        num_example_samples=args.num_example_samples,
        limit_run=args.limit_run,
    )

    solutions = {}
    for problem_id, problem in tqdm.tqdm(dataset, "Solve"):
        tests_solutions = []
        for test in problem["test"]:
            attempts = {}
            for attempt in range(1, 3):
                sequence, *_ = arc.transform.sample_sequence(
                    examples=problem["train"],
                    augment=augment,
                    tokenizer=tokenizer,
                    max_length=30 * 31 * 3 + 7,
                    test=test,
                    generate_test=False,
                    return_img_sequence=False,
                )
                solution = solve_problem(
                    examples=problem["train"],
                    test=test,
                    tokenizer=tokenizer,
                    model=model,
                    num_samples=args.num_samples,
                    device=device,
                )
                attempts[f"attempt_{attempt}"] = solution.tolist()
            tests_solutions.append(attempts)
        solutions[problem_id] = tests_solutions
    with open(args.output, "w") as submission_file:
        json.dump(solutions, submission_file)

    if args.solutions:
        scores = {}
        for problem_id, problem in tqdm.tqdm(dataset, "Score"):
            scores[problem_id] = []
            for t, test in enumerate(problem["test"]):
                solution = solutions[problem_id][t]
                expected = test["output"].tolist()
                scores[problem_id].append(
                    expected == solution["attempt_1"]
                    or expected == solution["attempt_2"]
                )
        max_score = sum(len(s) for s in scores.values())
        per_problem_scores = {p: sum(s) for p, s in scores.items()}
        total_score = sum(s for s in per_problem_scores.values())
        print(
            json.dumps(
                {p: s for p, s in per_problem_scores.items() if s != 0},
                indent=4,
            )
        )
        print(f"{total_score=}/{max_score=}")


if __name__ == "__main__":
    main()
