from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
# import lerobot.common.datasets.lerobot_dataset as lerobot_dataset # TODO: figure out why what's wrong with this import
import numpy as np
import torch
import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms
from torch.utils.data import ConcatDataset, WeightedRandomSampler, Sampler


T_co = TypeVar("T_co", covariant=True)

class MixtureSampler(Sampler[int]):
    def __init__(self, lengths, probs, num_samples, generator=None):
        """
        lengths: list[int]   # lengths per sub-dataset in ConcatDataset order
        probs:   list[float] # target per-dataset probabilities; need not sum exactly to 1
        num_samples: int     # draws per epoch
        """
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.cum = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.long)
        p = torch.tensor(probs, dtype=torch.float)
        self.probs = p / p.sum()
        self.num_samples = int(num_samples)
        self.gen = generator or torch.Generator()

    def __iter__(self):
        # probs/lengths should live on CPU for the main-process sampler
        probs_cpu   = self.probs.float().cpu()
        lengths_cpu = self.lengths.cpu()
        cum_cpu     = self.cum.cpu()

        for _ in range(self.num_samples):
            # pick which dataset k to sample from
            k = int(torch.multinomial(
                probs_cpu, 1, replacement=True, generator=self.gen
            ).item())

            n_k = int(lengths_cpu[k].item())
            if n_k <= 0:
                # if a dataset is empty, resample k
                # (fast path; extremely rare unless a dataset is length 0)
                while n_k <= 0:
                    k = int(torch.multinomial(
                        probs_cpu, 1, replacement=True, generator=self.gen
                    ).item())
                    n_k = int(lengths_cpu[k].item())

            # sample local index j in [0, n_k)
            j = int(torch.randint(n_k, (1,), generator=self.gen).item())

            # global index in ConcatDataset
            yield int(cum_cpu[k].item() + j)

    def __len__(self):
        return self.num_samples

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # TODO: set overlay options for keypoint drawing on image:
    overlay = True

    # Lazy import to avoid hard dependency unless used
    from openpi.training.egodex_dataset import EgoDexSeqDataset
    egodex_root = "/iris/projects/humanoid/dataset/ego_dex"
    egodex_dataset = EgoDexSeqDataset(
        root_dir=str(egodex_root),
        action_horizon=action_horizon//2,                                # use model’s horizon
        image_size=getattr(data_config, "image_size", (224, 224)),
        # TODO: EGO SPLIT IS IMPORTANT, OTHERWISE CHECK STATE FORMAT!
        state_format=getattr(data_config, "state_format", "ego_split"),     # e.g., "ego" or dataset default
        window_stride=getattr(data_config, "window_stride", 1),
        traj_per_task=getattr(data_config, "traj_per_task", None),    # optional subsampling
        max_episodes=getattr(data_config, "max_episodes", None),      # optional cap
        rebuild_index=False,  # TODO: choose correctly
        ############# TODO: TODO TODO TODO: set load_images to TRUEEEE when training!
        load_images = True, # TODO: choose correctly # second TODO: set to true to start training
        overlay = overlay)

    # TODO: uncomment below!
    from openpi.training.galaxea_dataset import GalaxeaDatasetKeypointsJoints
    dataset_dir1 = "/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/banana"
    dataset_dir2 = "/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/long_green_cube"
    dataset_dir3 = "/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/small_blue_cube"
    dataset_dir4 = "/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/yellow_duck"
    dataset_dir5 = "/iris/projects/humanoid/dataset/New_QUEST_DATA_ROBOT"


    # TODO: add more datasets for galaxea
    gd1 = GalaxeaDatasetKeypointsJoints(task = "vertical_pick_place", dataset_dir = dataset_dir1,
                                chunk_size=action_horizon//2, stride = 3, # TODO: change stride as needed
                                overlay = overlay) # TODO: action horizon // 2 is important if interleaving left and right actions

    gd2 = GalaxeaDatasetKeypointsJoints(task = "vertical_pick_place", dataset_dir = dataset_dir2,
                                chunk_size=action_horizon//2, stride = 3, # TODO: change stride as needed
                                overlay = overlay) # TODO: action horizon // 2 is important if interleaving left and right actions

    gd3 = GalaxeaDatasetKeypointsJoints(task = "vertical_pick_place", dataset_dir = dataset_dir3,
                                chunk_size=action_horizon//2, stride = 3, # TODO: change stride as needed
                                overlay = overlay) # TODO: action horizon // 2 is important if interleaving left and right actions

    gd4 = GalaxeaDatasetKeypointsJoints(task = "vertical_pick_place", dataset_dir = dataset_dir4,
                                chunk_size=action_horizon//2, stride = 3, # TODO: change stride as needed
                                overlay = overlay) # TODO: action horizon // 2 is important if interleaving left and right actions

    gd5 = GalaxeaDatasetKeypointsJoints(task = "assemble_quest", dataset_dir = dataset_dir5,
                                chunk_size=action_horizon//2, stride = 3, # TODO: change stride as needed
                                overlay = overlay) # TODO: action horizon // 2 is important if interleaving left and right actions

    return egodex_dataset, gd1, gd2, gd3, gd4, gd5

def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        print("repo id:", data_config.repo_id)
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
        print("HEY I've loaded the normalization stats! for repo:", data_config.repo_id)

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config2 = config.data2.create(config.assets_dirs, config.model) # TODO: clean this design

    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config, data_config2,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    data_config2: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """

    egodex_dataset, ds1, ds2, ds3, ds4, ds5 = create_torch_dataset(data_config, action_horizon, model_config)
    egodex_dataset = transform_dataset(egodex_dataset, data_config, skip_norm_stats=skip_norm_stats)
    # NOTE I'm using dataconfig2 here
    ds1 = transform_dataset(ds1, data_config2, skip_norm_stats=skip_norm_stats)
    ds2 = transform_dataset(ds2, data_config2, skip_norm_stats=skip_norm_stats)
    ds3 = transform_dataset(ds3, data_config2, skip_norm_stats=skip_norm_stats)
    ds4 = transform_dataset(ds4, data_config2, skip_norm_stats=skip_norm_stats)
    ds5 = transform_dataset(ds5, data_config2, skip_norm_stats=skip_norm_stats)

    print("I've loaded and transformed all datasets!")

    # 1) pick target mix; must sum to 1
    # TODO: ensure distribution looks good
    target_p = {
        "egodex": 0.75,
        "ds1":     0.05,
        "ds2":     0.05,
        "ds3":     0.05,
        "ds4":     0.05,
        "ds5":     0.05,
    }

    lengths = [len(egodex_dataset), len(ds1), len(ds2), len(ds3), len(ds4), len(ds5)]
    pks     = [target_p["egodex"],   target_p["ds1"],  target_p["ds2"], target_p["ds3"], target_p["ds4"], target_p["ds5"]]

    # 2) per-sample weights: w_k ∝ p_k / n_k
    w = []
    for n_k, p_k in zip(lengths, pks):
        w_k = (p_k / max(1, n_k))
        w.extend([w_k] * n_k)

    weights = torch.as_tensor(np.array(w, dtype=np.float64))

    # 3) choose an "epoch" length (how many draws per epoch)
    epoch_samples = sum(lengths)  # or any value you like

    # TODO: uncomment if co-training on all datasets
    dataset = ConcatDataset([egodex_dataset, ds1, ds2, ds3, ds4, ds5])
    # TODO: if using a single dataset
    # dataset = galaxea_dataset
    # dataset = ConcatDataset([egodex_dataset, galaxea_dataset])
    
    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
    
    # TODO: use me if training on a single dataset
    # sampler = None

    # TODO: use me if using multi-dataset
    sampler = MixtureSampler(
        lengths=lengths,
        probs=pks,                 # your target_p values in the same order as lengths
        num_samples=epoch_samples,
        generator=torch.Generator().manual_seed(seed),
    )


    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        # TODO: make shuffle false once using sampler
        shuffle=False, # (sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    # this code converts your image 0 - 255 into float -1 to 1
    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
