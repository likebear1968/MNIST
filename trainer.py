import numpy as np

class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer

    def fit(self, x, t, epoch_size, batch_size):
        size = len(x)
        iters = size // batch_size
        for ep in range(epoch_size):
            total_loss = 0
            loss_count = 0
            accuracy = 0
            for it in range(iters):
                mask = np.random.choice(size, batch_size)
                x_batch, t_batch = x[mask], t[mask]
                y = self.model.predict(x_batch, True)
                loss = self.model.loss.forward(y, t_batch)
                self.model.backward()
                self.optimizer.update(self.model.params, self.model.grads)
                total_loss += loss
                loss_count += 1
                if t_batch.ndim != 1: t_batch = np.argmax(t_batch, axis=1)
                accuracy += np.sum(np.argmax(y, axis=1) == t_batch) / float(x_batch.shape[0])
            print('loss %.2f | accuracy %.4f' %(total_loss / loss_count, accuracy / loss_count))
