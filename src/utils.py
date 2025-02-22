import torch


def batchify(data: torch.Tensor, batch_size):
    # Tính số batch trên data
    num_batches = data.size(0) // batch_size
    # Lấy đủ số lượng batch có thể lấy trên dữ liệu và cắt bỏ những dữ liệu cuối
    data = data.narrow(0, 0, num_batches * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(num_batches, -1).t().contiguous()
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