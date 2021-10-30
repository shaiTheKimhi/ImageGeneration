


PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=8,
        h_dim=32, z_dim=8, x_sigma2=0.0005,
        learn_rate=0.0001, betas=(0.9, 0.999),
    )

    hypers = {
        'batch_size': 8,
        'h_dim': 512,
        'z_dim': 32,
        'x_sigma2': 0.0015,
        'learn_rate': 0.0001,
        'betas': (0.9, 0.999)
    }
    return hypers



PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            
        ),
    )
    adam_d = {'type': 'Adam', 'weight_decay': 0.01, 'betas': (0.5, 0.99), 'lr': 0.0002}
    adam_g = {'type': 'Adam', 'weight_decay': 0.01, 'betas': (0.5, 0.99), 'lr': 0.0001}

    hypers = dict(
        batch_size=32, z_dim=128,
        data_label=1, label_noise=0.2,
        discriminator_optimizer=adam_d,
        generator_optimizer=adam_g
    )
    
    return hypers


