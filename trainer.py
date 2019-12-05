import numpy as np

class Trainer:
    def __init__(self, model, optimizer, metrics):
        self.model, self.optimizer, self.metrics = model, optimizer, metrics

    def fit(self, x, t, epoch_size, batch_size):
        size = len(x)
        iters = size // batch_size
        for ep in range(epoch_size):
            total_loss = 0
            loss_count = 0
            evaluate = np.array([])
            #evaluate = 0
            for it in range(iters):
                mask = np.random.choice(size, batch_size)
                x_batch, t_batch = x[mask], t[mask]
                y = self.model.predict(x_batch, True)
                loss = self.model.loss.forward(y, t_batch)
                self.model.backward()
                self.optimizer.update(self.model.params, self.model.grads)
                total_loss += loss
                loss_count += 1
                evaluate = np.append(evaluate, self.metrics.evaluate(t_batch, self.model.loss.y))
                #evaluate += self.metrics.evaluate(t_batch, self.model.loss.y)
            evaluate = np.sum(evaluate.reshape(-1, 4), axis=0) / loss_count
            print('lss %.2f | acc %.4f | f1 %.4f' %(total_loss / loss_count, evaluate[0], evaluate[3]))
            #print('loss %.2f | accuracy %.4f' %(total_loss / loss_count, evaluate / loss_count))
