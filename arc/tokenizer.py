"""ARC problem tokenizer."""

import torch


class ARCTokenizer:
    def __init__(self, max_run_length: int, num_values: int = 10) -> None:
        if max_run_length < 0:
            raise ValueError("Max run length must be >= 1.")

        self.special_tokens = {
            token: value
            for value, token in enumerate(("<pad>", "<in>", "<out>", "<rhs>"))
        }
        self.max_run_length = max_run_length
        self.num_token_ids = num_values * max_run_length + len(
            self.special_tokens
        )

    @torch.no_grad
    def _rle_tokenize(
        self,
        input: torch.Tensor,
        max_run_length: int = -1,
        add_rhs_token: bool = False,
        remove_trailing_rhs_token: bool = True,
    ) -> None:
        """Tokenize based on the run length of the values. All values must be >= 0.

        For each value there are self.max_run_length tokens. The first
        self.max_run_length tokens ids are special and reserved tokens.
        The value tokens ids start self.max_run_length.
        """

        if input.dim() == 1:
            input = torch.unsqueeze(input, 0)

        rhs_value = -1
        if add_rhs_token:
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
        if (
            remove_trailing_rhs_token
            and tokens[-1] == self.special_tokens["<rhs>"]
        ):
            tokens = tokens[:-1]

        return tokens

    def encode(
        self,
        input: list[torch.Tensor],
        max_run_length: int = -1,
        add_special_tokens: bool = True,
        add_solution_prompt: bool = True,
    ) -> torch.Tensor:
        tokens = []
        in_token = torch.tensor([self.special_tokens["<in>"]])
        out_token = torch.tensor([self.special_tokens["<out>"]])
        for i, tensor in enumerate(input):
            if add_special_tokens:
                tokens.append(out_token if i % 2 else in_token)
            tokens.append(
                self._rle_tokenize(tensor, max_run_length, add_special_tokens)
            )

        if add_solution_prompt:
            tokens.append(torch.tensor(self.special_tokens["<out>"]))

        tokens = torch.hstack(tokens)
        return tokens

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        pass
