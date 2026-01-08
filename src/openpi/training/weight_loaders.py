import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

         # Flatten checkpoint params
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        # If your checkpoint stores everything under "params/...", strip it to match model keys
        if flat_loaded and all(isinstance(k, str) and k.startswith("params/") for k in flat_loaded):
            flat_loaded = {k[len("params/"):]: v for k, v in flat_loaded.items()}

        # Drop exactly the offending keys. Use a robust regex to handle optional prefixes.
        drop_re = re.compile(
            # r"(?:.*/)?(?:(?:action_(?:in|out)_proj/(?:kernel|bias))|(state_proj/kernel))$"
            r"(?:.*/)?action_(?:in|out)_proj/(?:kernel|bias)$"
        )   

        dropped = []
        for k in list(flat_loaded.keys()):
            if drop_re.fullmatch(k):
                dropped.append((k, getattr(flat_loaded[k], "shape", None)))
                del flat_loaded[k]

        if dropped:
            print("[CheckpointWeightLoader] Dropping keys so they re-init:")
            for k, shp in dropped:
                print("  DROP", k, "shape:", shp)
        # Rebuild nested dict
        loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")

        # print("[CheckpointWeightLoader] Loaded parameters:")
        # flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        # for k in sorted(flat_loaded.keys()):
        #     v = flat_loaded[k]
        #     print(f"  {k}: shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")
        # assert False

        # original
        # return _merge_params(loaded_params, params, missing_regex=".*lora.*")

        # Add all missing LoRA weights.
        # TODO: missing_regex of ".*" might be too general and remiss
        return _merge_params(loaded_params, params, missing_regex=".*") 

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")

def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/") # NOTE: loaded params do not contain the action projection layers, since we surgically removed them

    # Take weights whose keys exist AND shapes match; cast to ref dtype.
    result = {}
    for k, v in flat_loaded.items(): # go through loaded weight
        if k in flat_ref: # if loaded key exists in the model key
            ref = flat_ref[k] # get the model weight for this key
            if getattr(v, "shape", None) == getattr(ref, "shape", None):
                result[k] = v.astype(getattr(ref, "dtype", getattr(v, "dtype", None)))
            # else: skip; will be backfilled below if allowed by regex

    # Backfill missing/skipped keys according to policy.
    pattern = re.compile(missing_regex)
    for k, ref in flat_ref.items():
        if k not in result and pattern.fullmatch(k):
            result[k] = ref

    return flax.traverse_util.unflatten_dict(result, sep="/")

# original merge_params
# def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
#     """Merges the loaded parameters with the reference parameters.

#     Args:
#         loaded_params: The parameters to merge.
#         params: The reference parameters.
#         missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

#     Returns:
#         A new dictionary with the merged parameters.
#     """
#     flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
#     flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

#     # First, take all weights that are a subset of the reference weights.
#     result = {}
#     for k, v in flat_loaded.items():
#         if k in flat_ref:
#             result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

#     flat_loaded.clear()

#     # Then, merge any missing weights as defined by the missing regex.
#     pattern = re.compile(missing_regex)
#     for k in {k for k in flat_ref if pattern.fullmatch(k)}:
#         if k not in result:
#             result[k] = flat_ref[k]

#     return flax.traverse_util.unflatten_dict(result, sep="/")
