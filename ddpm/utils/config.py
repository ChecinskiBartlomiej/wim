class CONFIG:
    # Relative paths (run from home directory on supercomputer)
    model_path = "outputs/ddpm_unet.pth"
    train_csv_path = "data/MNIST/train.csv"
    test_csv_path = "data/MNIST/test.csv"
    generated_csv_path = "outputs/mnist_generated_data.csv"
    num_epochs = 50
    lr = 1e-4
    num_timesteps = 1000
    batch_size = 128
    img_size = 28
    in_channels = 1
    num_img_to_generate = 256
