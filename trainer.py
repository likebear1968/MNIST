import numpy as np

class Trainer:
    def __init__(self, model, optimizer, metrics):
        self.model, self.optimizer, self.metrics = model, optimizer, metrics

    def fit(self, x, t, epoch_size, batch_size):
        size = len(x)
        iters = size // batch_size
        for ep in range(epoch_size):
            for it in range(iters):
                mask = np.random.choice(size, batch_size)
                x_batch, t_batch = x[mask], t[mask]
                y = self.model.predict(x_batch, True)
                loss, prd = self.model.loss.forward(y, t_batch)
                self.model.backward()
                self.optimizer.update(self.model.params, self.model.grads)
                self.metrics.add_metrics(loss, t_batch, prd)
            self.metrics.get_metrics()
