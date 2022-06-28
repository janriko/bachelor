"""
Training parameters
"""
num_epochs = 2


def num_training_steps(dataloader) -> int:
    return num_epochs * len(dataloader)
