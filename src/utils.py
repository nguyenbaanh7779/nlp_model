import torch


def batchify(data: torch.Tensor, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nb_batches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nb_batches * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


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