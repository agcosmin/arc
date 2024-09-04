"""ARC challenge dataset"""

import json
import typing
import itertools

import torch
import torch.utils.data

import arc.transform


class ARCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        problems_sets: list[tuple[str, typing.Optional[str]]],
    ) -> None:
        super().__init__()
        self.problems = {}
        for problems_path, solutions_path in problems_sets:
            with open(problems_path, "r") as problems_file:
                problems = json.load(problems_file)

            if solutions_path:
                with open(solutions_path, "r") as solutions_file:
                    solutions = json.load(solutions_file)

                for pid in problems:
                    for i in range(len(problems[pid]["test"])):
                        problems[pid]["test"][i]["output"] = solutions[pid][i]

            self.problems.update(problems)
        self.problems_ids = [pid for pid in self.problems]

    def __len__(self):
        return len(self.problems_ids)

    def __getitem__(self, key: int) -> tuple[str, arc.transform.ARCProblem]:
        problem_id = self.problems_ids[key]
        return problem_id, arc.transform.problem_to_tensor(
            self.problems[problem_id]
        )


class ARCCollator:
    def __init__(self, pad: int, generate_attention_mask=True) -> None:
        self.pad = pad
        self.generate_attention_mask = generate_attention_mask

    def __call__(
        self,
        items: typing.Union[
            list[list[dict[str, typing.Any]]], list[dict[str, typing.Any]]
        ],
    ) -> dict[str, typing.Any]:
        if items and isinstance(items[0], list):
            items = list(itertools.chain.from_iterable(items))
        problems = [item["problem"] for item in items]
        sequences = [item["tokenized_sequence"] for item in items]
        max_seq_len = max([len(seq) for seq in sequences])
        input_ids = torch.full((len(sequences), max_seq_len), self.pad)
        key_padding_mask = torch.zeros((len(sequences), max_seq_len))
        for i, sequence in enumerate(sequences):
            input_ids[i, 0 : len(sequence)] = sequence
            key_padding_mask[i, len(sequence) :] = -torch.inf

        batch = {
            "problem": problems,
            "input_ids": input_ids,
            "padding_mask": key_padding_mask,
        }
        return batch
