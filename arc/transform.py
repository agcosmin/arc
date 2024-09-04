"""ARC problem transform."""

import bisect
import dataclasses
import typing
import itertools
import arc.tokenizer

import torch


ARCExample = dict[str, typing.Union[list[list[int]], torch.Tensor]]
ARCProblem = dict[str, list[ARCExample]]


@dataclasses.dataclass
class Augment:
    value_permutation: bool = False
    fliplr: bool = False
    flipud: bool = False
    rotate: bool = False
    num_example_samples: int = -1
    limit_run: bool = False


def problem_to_tensor(problem: ARCProblem) -> ARCProblem:
    problem = {
        phase: [
            {key: torch.tensor(value) for key, value in pair.items()}
            for pair in pairs
        ]
        for phase, pairs in problem.items()
    }
    return problem


@torch.no_grad
def permute_values(
    input: torch.Tensor, value_permutation: list[int]
) -> torch.Tensor:
    output = torch.clone(input)
    for from_value, to_value in enumerate(value_permutation, 1):
        output[input == from_value] = to_value

    return output


@dataclasses.dataclass
class Transform:
    value_permutation: typing.Optional[torch.Tensor] = None
    fliplr: bool = False
    flipud: bool = False
    rotate: int = -1

    @torch.no_grad
    def __call__(self, example: ARCExample) -> ARCExample:
        if self.value_permutation is not None:
            example = {
                k: permute_values(v, self.value_permutation)
                for k, v in example.items()
            }
        if self.fliplr:
            example = {k: torch.fliplr(v) for k, v in example.items()}
        if self.flipud:
            example = {k: torch.flipud(v) for k, v in example.items()}
        if self.rotate > 0:
            example = {
                k: torch.rot90(v, self.rotate) for k, v in example.items()
            }

        return example


def generate_transform(
    examples: list[ARCExample],
    value_permutation: bool = True,
    fliplr: bool = True,
    flipud: bool = True,
    rotate: bool = True,
    num_example_samples: int = -1,
    limit_run: bool = True,
) -> Transform:
    if num_example_samples > 0:
        samples = torch.multinomial(
            torch.full((len(examples),), 1 / len(examples)),
            num_samples=num_example_samples,
            replacement=True,
        )
    else:
        samples = torch.arange(len(examples))

    transforms = [
        (
            sample,
            Transform(
                torch.randperm(9) + 1 if value_permutation else None,
                fliplr and bool(torch.randint(0, 2, (1,))),
                flipud and bool(torch.randint(0, 2, (1,))),
                int(rotate) * torch.randint(0, 4, (1,)).item(),
            ),
        )
        for sample in samples
    ]

    if limit_run:
        min_width = min(min(len(v[0]) for v in ex.values()) for ex in examples)
        run_limit = torch.randint(1, min_width + 1, (1,)).item()
    else:
        run_limit = -1

    return transforms, run_limit


def sample_sequence(
    examples: list[dict[str, torch.Tensor]],
    augment: Augment,
    tokenizer: arc.tokenizer.ARCTokenizer,
    max_length: int = -1,
    test: typing.Optional[dict[str, torch.Tensor]] = None,
    generate_test: bool = False,
    return_img_sequence: bool = False,
) -> tuple[
    torch.Tensor,
    typing.Optional[dict[str, torch.Tensor]],
    typing.Optional[list[torch.Tensor]],
]:
    transforms, run_limit = generate_transform(
        examples,
        augment.value_permutation,
        augment.fliplr,
        augment.flipud,
        augment.rotate,
        augment.num_example_samples,
        augment.limit_run,
    )
    examples = [transform(examples[i]) for i, transform in transforms]
    if test is None and generate_test:
        *examples, test = examples
    sequence = list(
        itertools.chain.from_iterable(
            [[ex["input"], ex["output"]] for ex in examples]
        )
    )
    if test:
        sequence.append(test["input"])

    token_sequence = tokenizer.encode(
        sequence,
        max_run_length=run_limit,
        add_solution_prompt=bool(test),
    )

    if max_length > 0 and len(token_sequence) > max_length:
        boi_pos = token_sequence == tokenizer.special_tokens["<boi>"]
        boi_pos = boi_pos.nonzero().flatten()
        lengths = (len(token_sequence) - boi_pos).flip(dims=(0,)).tolist()
        cut_idx = bisect.bisect_right(lengths, max_length)
        if cut_idx == 0:
            raise RuntimeError("Could no find cut point.")
        else:
            cut_point = len(token_sequence) - lengths[cut_idx - 1]
            token_sequence = token_sequence[cut_point:]

    return (
        token_sequence,
        test,
        sequence if return_img_sequence else None,
    )
