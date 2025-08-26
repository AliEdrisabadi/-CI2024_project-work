import numpy as np
import matplotlib.pyplot as plt

# saved a numpy array with best-MSE per generation, load and plot it.

hist = np.load("./plots/history_problem_1.npy")
plt.figure(figsize=(7,4))
plt.plot(hist)
plt.title("Best MSE per Generation (problem 1)")
plt.xlabel("Generation"); plt.ylabel("Best MSE")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("./plots/history_problem_1_curve.png", dpi=160)
