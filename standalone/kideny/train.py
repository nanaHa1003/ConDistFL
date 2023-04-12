import os
import json

import numpy as np
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from custom.data import DataManager
from custom.trainer import Trainer
from custom.validator import Validator
from custom.utils.get_model import get_model

if __name__ == "__main__":
    app_root = os.getcwd()

    with open("config/config_task.json") as f:
        task_config = json.load(f)

    with open("config/config_data.json") as f:
        data_config = json.load(f)

    # Create data & model
    dm = DataManager(app_root, data_config)
    model = get_model(task_config["model"]).to("cuda:0")

    best_metric = -np.inf

    # Create trainer & validator
    trainer = Trainer(task_config)
    validator = Validator(task_config)

    # Create tensorboard logger
    logger = SummaryWriter(log_dir="logs")

    dm.setup("train")
    dm.setup("validate")

    # Run round based training & validation
    for i in range(240):
        trainer.run(
            model,
            dm.get_data_loader("train"),
            num_steps=500,
            logger=logger
        )
        metrics = validator.run(model, dm.get_data_loader("validate"))

        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for m, v in metrics.items():
            table.add_row([m, v])
            logger.add_scalar(m, v, i)
        print(table)

        if best_metric < metrics["val_meandice"]:
            best_metric = metrics["val_meandice"]
            trainer.save_checkpoint("models/best_model.ckpt", model)
    # Save last checkpoint
    trainer.save_checkpoint("models/last.ckpt", model)

    # Load best checkpoint
    ckpt = torch.load("models/best_model.ckpt")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to("cuda:0")

    dm.setup("test")
    metrics = validator.run(model, dm.get_data_loader("test"))
    print(f"Test metrics: {metrics}")

