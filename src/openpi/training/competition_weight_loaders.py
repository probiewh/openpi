import dataclasses

import numpy as np

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class HistoryCheckpointWeightLoader(_weight_loaders.WeightLoader):
    """Load a Pi0.5 checkpoint while initializing new history parameters.

    The released checkpoint has no temporal projection parameters. This loader
    preserves their freshly initialized values in the copied history model while
    still loading every matching base and LoRA-compatible parameter.
    """

    params_path: str

    def load(self, params):
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        return _weight_loaders._merge_params(  # noqa: SLF001
            loaded_params,
            params,
            missing_regex=".*(lora|history).*",
        )
