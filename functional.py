import numpy as np


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0, keepdims=True)


def cross_entropy(inp, targ):
    p = softmax(inp)
    log_likelihood = -np.log(p[range(inp.shape[0]), targ.argmax(axis=1)] + 1e-9)
    loss = np.sum(log_likelihood) / inp.shape[0]
    return loss


def accuracy(inp, targ):
    pred = softmax(inp)
    return (np.argmax(pred, axis=1) == targ).astype(float).mean()


def regions_for_conv2d(inp, kernel_size=3, padding=1, stride=1):
    bs = inp.shape[0]

    for b in range(bs):
        padded = np.copy(inp[b])
        padded = np.pad(padded, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        w, h, _ = padded.shape

        for i in range(0, w - kernel_size + 1, stride):
            for j in range(0, h - kernel_size + 1, stride):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                yield b, region, i, j


def conv2d(inp, filters, kernel_size=3, padding=1, stride=1):
    bs, w, h, c = inp.shape
    nb_filters = filters.shape[2]
    final_filters = nb_filters if c == 1 else nb_filters + c
    output = np.zeros((bs, w - kernel_size + 1 + 2 * padding, h - kernel_size + 1 + 2 * padding, final_filters))

    for b, region, i, j in regions_for_conv2d(inp, kernel_size, padding, stride):
        output[b, i, j] = np.sum(region * filters)

    return output


def conv2d_backprop(inp, out_grad, filters, kernel_size=3, padding=1, stride=1):
    filters_grad = np.zeros(filters.shape)
    inp_grad = np.zeros(inp.shape)
    inp_grad_padded = np.copy(inp_grad)
    inp_grad_padded = np.pad(inp_grad_padded, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    nb_filters = filters.shape[2]

    for b, region, i, j in regions_for_conv2d(inp, kernel_size, padding, stride):
        for f in range(nb_filters):
            inp_grad_padded[b, i:i + kernel_size, j:j + kernel_size, 0] += filters[:, :, f] * out_grad[b, i, j, f]
            filters_grad[:, :, f] += np.squeeze(region, axis=2) * out_grad[b, i, j, f]

    inp_grad[:, :, :, :] = inp_grad_padded[:, padding: -padding, padding: -padding, :]

    return filters_grad, inp_grad


def regions_for_maxpool2d(inp):
    bs, w, h, _ = inp.shape
    new_w, new_h = w // 2, h // 2

    for b in range(bs):
        for i in range(new_w):
            for j in range(new_h):
                region = inp[b, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
                yield b, region, i, j


def maxpool2d(inp):
    bs, w, h, c = inp.shape
    output = np.zeros((bs, w // 2, h // 2, c))

    for b, region, i, j in regions_for_maxpool2d(inp):
        output[b, i, j] = np.amax(region)

    return output


def maxpool2d_backprop(inp, out_grad):
    inp_grad = np.zeros(inp.shape)

    for b, region, i, j in regions_for_maxpool2d(inp):
        w, h, f = region.shape
        amax = np.amax(region, axis=(0, 1))

        for i2 in range(w):
            for j2 in range(h):
                for f2 in range(f):
                    if region[i2, j2, f2] == amax[f2]:
                        inp_grad[b, i * 2 + i2, j * 2 + j2, f2] = out_grad[b, i, j, f2]

    return inp_grad