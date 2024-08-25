"""ARC challenge dataset"""

import bisect
import json
import typing
import itertools

import torch
import torch.utils.data

import arc.transform
import arc.tokenizer


class ARCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        problems_sets: list[tuple[str, typing.Optional[str]]],
        max_sequence_length=30 * 31 * 4 + 5,  # 2 example of max size w/ run 1
        train: bool = True,
        tokenizer: typing.Optional[arc.tokenizer.ARCTokenizer] = None,
        augment: typing.Optional[arc.transform.Augment] = None,
        return_raw: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.augment = augment
        self.return_raw = return_raw
        self.train = train
        self.max_sequence_length = max_sequence_length
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

    def __getitem__(self, key: int) -> dict[str, typing.Any]:
        problem_id = self.problems_ids[key]
        problem = arc.transform.problem_to_tensor(self.problems[problem_id])
        examples = problem["train"] + (problem["test"] if self.train else [])
        run_limit = -1
        if self.augment:
            transforms, run_limit = arc.transform.generate_transform(
                examples,
                self.augment.value_permutation,
                self.augment.fliplr,
                self.augment.flipud,
                self.augment.rotate,
                self.augment.num_example_samples,
                self.augment.limit_run,
            )
            examples = [transform(examples[i]) for i, transform in transforms]

        sequence = sum([[ex["input"], ex["output"]] for ex in examples], [])
        if self.train:
            sequences = [sequence]
        else:
            sequences = [sequence + [test["input"]] for test in problem["test"]]

        pruned_tokenized_sequences = []
        if self.tokenizer:
            tokenized_sequences = [
                self.tokenizer.encode(
                    seq,
                    max_run_length=run_limit,
                    add_special_tokens=True,
                    add_solution_prompt=not self.train,
                )
                for seq in sequences
            ]
            for i, sequence in enumerate(tokenized_sequences):
                if (
                    self.max_sequence_length > 0
                    and len(sequence) > self.max_sequence_length
                ):
                    in_pos = (
                        (sequence == self.tokenizer.special_tokens["<in>"])
                        .nonzero()
                        .flatten()
                    )
                    lengths = (len(sequence) - in_pos).flip(dims=(0,)).tolist()
                    cut_idx = bisect.bisect_right(
                        lengths, self.max_sequence_length
                    )
                    if cut_idx == 0:
                        print(
                            f"Warning: Could not find cut point for sequence from {problem_id}."
                        )
                    else:
                        cut_point = len(sequence) - lengths[cut_idx - 1]
                        pruned_tokenized_sequences.append(sequence[cut_point:])
                else:
                    pruned_tokenized_sequences.append(sequence)

        items = []
        for i in range(len(sequences)):
            item = {"problem": problem_id}
            if pruned_tokenized_sequences:
                item["tokenized_sequence"] = pruned_tokenized_sequences[i]
            if self.return_raw:
                item["sequence"] = sequence[i]
            items.append(item)

        return items


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
        key_padding_mask = torch.zeros(
            (len(sequences), max_seq_len), dtype=bool
        )
        for i, sequence in enumerate(sequences):
            input_ids[i, 0 : len(sequence)] = sequence
            key_padding_mask[i, len(sequence) :] = True

        batch = {
            "problem": problems,
            "input_ids": input_ids,
            "padding_mask": key_padding_mask,
        }
        return batch
