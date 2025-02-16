import torch
import numpy as np
import yaml
import json
import datasets
from torch import cuda
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
)

from src.data.preparing import CustomDataset


def train_each_epoch(epoch, model, device, loader, optimizer):
    logging = {
        "optimize_loss": [],
        "encoder_z_loss": [],
        "encoder_aux_loss": [],
        "decoder_z_loss": [],
        "decoder_aux_loss": [],
    }
    model.train()
    for _, data in enumerate(loader, 0):
        labels = data["target_ids"].to(device, dtype=torch.long)
        labels = model._shift_right(labels)

        # We set the pad tokens (0) to -100 to be
        # ignored by the CrossEntropy loss
        labels = labels.masked_fill_(labels == 0, -100)
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            output_router_logits=True,
            return_dict=True,
        )

        loss = outputs[0]

        # logging loss during training model
        logging["optimize_loss"] = loss.item()
        logging["encoder_z_loss"] = outputs.encoder_z_loss.item()
        logging["encoder_aux_loss"] = outputs.encoder_aux_loss.item()
        logging["decoder_z_loss"] = outputs.decoder_z_loss.item()
        logging["decoder_aux_loss"] = outputs.decoder_aux_loss.item()

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        if (_ + 1) % 2000 == 0:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return model, logging


def train(model, device, training_loader, optimizer, num_epochs):
    # Training loop
    all_logging = dict()
    print("Initiating Fine-Tuning for the model on our dataset")
    for epoch in range(num_epochs):
        model, logging = train_each_epoch(
            epoch, model, device, training_loader, optimizer
        )
        all_logging[f"epoch_{epoch}"] = logging
    return model, all_logging


def run():
    with open("config/switch_transfomer/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    f.close()
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config["SEED"])  # pytorch random seed
    np.random.seed(config["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Prepare data to training
    print("Preparing data ...")
    dataset = datasets.load_from_disk("data/raw/xsum")
    # Creation of Dataset and Dataloader
    train_dataset = dataset["train"]
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    training_set = CustomDataset(
        train_dataset, tokenizer, config["MAX_LEN"], config["SUMMARY_LEN"]
    )
    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": config["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }
    training_loader = DataLoader(training_set, **train_params)

    # Initial device
    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Train model by {device}")
    # Initial model
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        "google/switch-base-8", torch_dtype=torch.bfloat16
    )
    model = model.to(device)
    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=float(config["LEARNING_RATE"])
    )

    # Start training model
    print("Training model ...")
    model, logging = train(
        model=model,
        device=device,
        training_loader=training_loader,
        optimizer=optimizer,
        num_epochs=config["TRAIN_EPOCHS"],
    )
    model.save_pretrained("model/switch_transformer")
    tokenizer.save_pretrained("tokenizer/switch_transformer")
    with open("result/switch_transformer/logging/loss_log.json", "r") as f:
        json.dump(logging, f)
    f.close()