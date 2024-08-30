"""ARC problem tokenizer."""

import torch


class ARCTokenizer:
    def __init__(self, max_run_length: int, num_values: int = 10) -> None:
        if max_run_length < 0:
            raise ValueError("Max run length must be >= 1.")

        self.special_tokens = {
            token: value
            for value, token in enumerate(
                ("<pad>", "<boi>", "<eoi>", "<boo>", "<eoo>", "<rhs>")
            )
        }
        self.max_run_length = max_run_length
        self.num_token_ids = num_values * max_run_length + len(
            self.special_tokens
        )

    @torch.no_grad
    def _rle_tokenize(
        self, input: torch.Tensor, max_run_length: int = -1
    ) -> None:
        """Tokenize based on the run length of the values. All values must be >= 0.

        For each value there are self.max_run_length tokens.
        """

        if input.dim() == 1:
            input = torch.unsqueeze(input, 0)

        rhs_value = -1
        rhs = torch.full((input.shape[0], 1, *input.shape[2:]), rhs_value)
        input = torch.hstack((input, rhs))

        if max_run_length > 0:
            max_run_length = min(max_run_length, self.max_run_length)
        else:
            max_run_length = self.max_run_length

        values, runs = torch.unique_consecutive(input, return_counts=True)
        if torch.any(runs > max_run_length):
            num_max_length = runs // max_run_length
            remainder = runs % max_run_length
            new_values = []
            new_runs = []
            for val, num_max, remain in zip(values, num_max_length, remainder):
                if num_max > 0:
                    new_values.extend([val] * num_max)
                    new_runs.extend([max_run_length] * num_max)
                if remain > 0:
                    new_values.append(val)
                    new_runs.append(remain)
            values = torch.stack(new_values)
            runs = torch.tensor(new_runs)

        tokens = (
            values * self.max_run_length + (runs - 1) + len(self.special_tokens)
        )
        tokens[values == rhs_value] = self.special_tokens["<rhs>"]

        return tokens

    def encode(
        self,
        input: list[torch.Tensor],
        max_run_length: int = -1,
        add_solution_prompt: bool = False,
    ) -> torch.Tensor:
        tokens = []
        bounds_tokens = (
            (
                torch.tensor([self.special_tokens["<boi>"]]),
                torch.tensor([self.special_tokens["<eoi>"]]),
            ),
            (
                torch.tensor([self.special_tokens["<boo>"]]),
                torch.tensor([self.special_tokens["<eoo>"]]),
            ),
        )
        for i, tensor in enumerate(input):
            prefix, suffix = bounds_tokens[i % 2]
            tokens.append(prefix)
            tokens.append(self._rle_tokenize(tensor, max_run_length))
            tokens.append(suffix)

        if add_solution_prompt:
            tokens.append(bounds_tokens[1][0])

        tokens = torch.hstack(tokens)
        return tokens

    def decode(self, sequence: torch.Tensor) -> list[torch.Tensor]:
        bound_marks = (
            torch.logical_or(
                torch.logical_or(
                    sequence == self.special_tokens["<boi>"],
                    sequence == self.special_tokens["<eoi>"],
                ),
                torch.logical_or(
                    sequence == self.special_tokens["<boo>"],
                    sequence == self.special_tokens["<eoo>"],
                ),
            )
            .nonzero()
            .flatten()
            .tolist()
        )
        rhs_mask = (sequence == self.special_tokens["<rhs>"]).tolist()
        sequence = sequence - len(self.special_tokens)
        values = torch.div(sequence, self.max_run_length, rounding_mode="trunc")
        runs = torch.remainder(sequence, self.max_run_length) + 1

        images = []
        for i in range(0, len(bound_marks), 2):
            begin, end = bound_marks[i] + 1, bound_marks[i + 1]
            if begin == end:
                continue
            rows = []
            row = []
            for value, run, rhs in zip(
                values[begin:end],
                runs[begin:end],
                rhs_mask[begin:end],
            ):
                if rhs:
                    rows.append(torch.hstack(row))
                    row = []
                else:
                    row.append(torch.full((run,), value))
            num_columns = max(len(row) for row in rows)
            img = torch.full((len(rows), num_columns), 0)
            for i, row in enumerate(rows):
                img[i, 0 : len(row)] = row
            images.append(img)
        return images
