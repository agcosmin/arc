"""ARC problem transform."""

import dataclasses
import typing

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
    for from_value, to_value in enumerate(value_permutation):
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
