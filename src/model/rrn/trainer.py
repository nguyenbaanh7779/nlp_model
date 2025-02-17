import torch
import time
import os
import yaml
import torch.nn as nn

import my_source.config_env as config_env
import src.utils as utils
from my_source.src.model.rrn.rnn import RNNModel


def get_batch(source, i, stride, evaluation=False):
    seq_len = min(stride, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)

    if evaluation:
        # Trong chế độ đánh giá, đảm bảo rằng không cần theo dõi gradient
        with torch.no_grad():
            data = data.clone()
            target = target.clone()

    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train_each_epoch(
    model,
    train_data,
    n_token,
    batch_size,
    criterion,
    learning_rate,
    stride,
    max_grad_norm,
    log_interval,
):
    start_time = time.time()
    total_loss = 0
    model.train()
    hidden = model.init_hidden(bsz=batch_size)

    if type(learning_rate) is str:
        learning_rate = float(learning_rate)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, stride)):
        data, target = get_batch(
            source=train_data, i=i, stride=stride
        )  # data(bptt, batch), target(bppt*batch)
        hidden = repackage_hidden(hidden)  # hidden(num_layer, batch_size, emb_size)

        model.zero_grad()
        output, hidden = model(
            data, hidden
        )  # output(bptt, batch_size, n_token), hidden(n_layer, batch_size, emb_size)
        loss = criterion(output.view(-1, n_token), target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        total_loss += loss

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                f"{batch}/{len(train_data)} batches | learning rate {learning_rate} | ms/batch {elapsed * 1000 / log_interval} | loss {cur_loss} | ppl {torch.exp(cur_loss)}"
            )


def train_model():
    # Init device
    print("Initaling device ...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load data
    print("Loading data ...")
    corpus_path = os.path.join(config_env.ROOT_PATH, "data/raw/pentext.pt")
    corpus = torch.load(corpus_path)

    # Init model
    print("Initaling model ...")
    with open(os.path.join(config_env.ROOT_PATH, "config/model_param.yaml"), "r") as f:
        model_param = yaml.safe_load(f)
    f.close()
    model_param["ntoken"] = corpus.vocab_size
    model = RNNModel(**model_param)

    # Train model
    print("Starting train model ...")
    with open(os.path.join(config_env.ROOT_PATH, "config/trainer_param.yaml"), "r") as f:
        trainer_params = yaml.safe_load(f)
    f.close()


    train_data = utils.batchify(
        data=corpus.train, batch_size=trainer_params["batch_size"]
    )
    # valid_data = utils.batchify(data=corpus.valid, batch_size=trainer_params["batch_size"])
    # test_data = utils.batchify(data=corpus.test, batch_size=trainer_params["batch_size"])

    trainer_params["model"] = model.to(device)
    trainer_params["train_data"] = train_data.to(device)
    trainer_params["n_token"] = corpus.vocab_size
    trainer_params["criterion"] = nn.CrossEntropyLoss()

    train_each_epoch(**trainer_params)


if __name__:
    train_model()