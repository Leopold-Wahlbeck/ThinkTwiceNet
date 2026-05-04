from src.data.loaders import DataConfig, load_cifar10_datasets, dataset_to_numpy


def main() -> None:
    config = DataConfig(
        data_dir="data/raw",
        val_size=0.15,
        random_state=42,
        flatten_for_stage1=True,
    )

    train_dataset, val_dataset, test_dataset = load_cifar10_datasets(
        data_dir=config.data_dir,
        val_size=config.val_size,
        random_state=config.random_state,
        flatten_for_stage1=config.flatten_for_stage1,
    )

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)


if __name__ == "__main__":
    main()