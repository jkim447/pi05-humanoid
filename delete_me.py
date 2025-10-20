# --- NEW: build a concatenated, per-dataset-normalized dataset with sampling weights ---

def _create_multi_torch_dataset_and_sampler(
    data_multi: _config.MultiDataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    *,
    skip_norm_stats: bool = False,
    assets_dirs: os.PathLike | str,
) -> tuple[ConcatDataset, WeightedRandomSampler]:
    """
    For each DataConfigFactory:
      - build its DataConfig (pulling its own norm_stats)
      - create its torch dataset
      - apply its own transforms + Normalize with its own stats
    Then concat and return a WeightedRandomSampler to mix between datasets.
    """
    datasets = []
    lengths = []
    for factory in data_multi.datasets:
        data_cfg = factory.create(assets_dirs, model_config)
        ds = create_torch_dataset(data_cfg, action_horizon, model_config)
        ds = transform_dataset(ds, data_cfg, skip_norm_stats=skip_norm_stats)
        datasets.append(ds)
        lengths.append(len(ds))

    concat = ConcatDataset(datasets)

    # Compute per-item weights based on per-dataset weights.
    if data_multi.weights is None:
        dataset_weights = np.ones(len(datasets), dtype=np.float64)
    else:
        dataset_weights = np.asarray(data_multi.weights, dtype=np.float64)
        dataset_weights = dataset_weights / dataset_weights.sum()

    # Expand dataset weights to per-sample weights
    per_sample_weights = []
    for w, n in zip(dataset_weights, lengths):
        # Weight per sample within a dataset is proportional to w / n
        # (so total expected share across the epoch is ~ w)
        per_sample_weights.extend([w / max(n, 1)] * n)
    per_sample_weights = torch.tensor(per_sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=per_sample_weights,
        num_samples=sum(lengths),  # one "epoch" worth; DataLoader will loop anyway
        replacement=True,
    )
    return concat, sampler
