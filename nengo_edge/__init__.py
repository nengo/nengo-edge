from typing_extensions import Protocol


class Runner(Protocol):
    def reset_state(self) -> None:
        ...

    def run(self, inputs: np.ndarray) -> np.ndarray:
        ...
