
import matplotlib.pyplot as plt

__all__ = ['ResultPlotter',]

class ResultPlotter:
    def __init__(self, init_method):
        self.init_method = init_method
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def plot_results(self, losses, accuracies):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(losses)
        self.ax1.set_title(f'Loss vs Epochs ({self.init_method})')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')

        self.ax2.plot(accuracies)
        self.ax2.set_title(f'Accuracy vs Epochs ({self.init_method})')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.show()

    def save_plot(self, filename_prefix):

        filename = f"{filename_prefix}_{self.init_method}.png"
        self.fig.savefig(filename)
        