
import matplotlib.pyplot as plt

__all__ = ['ResultPlotter',]

class ResultPlotter:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def plot_results(self, losses, accuracies, init_method):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(losses)
        self.ax1.set_title(f'Loss vs Epochs ({init_method})')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')

        self.ax2.plot(accuracies)
        self.ax2.set_title(f'Accuracy vs Epochs ({init_method})')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.show()

    def save_plot(self, filename):
        self.fig.savefig(filename)