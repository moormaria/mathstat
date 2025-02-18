import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import cauchy, norm, poisson, uniform

# define sample sizes
sample_sizes = [10, 100, 1000]

# define distributions
DISTRIBUTIONS = {
    "Normal": {
        "generator": norm.rvs,
        "density": norm.pdf,
        "x_values": np.linspace(-5, 5, 1000)
    },
    "Cauchy": {
        "generator": cauchy.rvs,
        "density": cauchy.pdf,
        "x_values": np.linspace(-10, 10, 1000)
    },
    "Poisson": {
        "generator": lambda size: poisson.rvs(mu=10, size=size),
        "density": lambda x: poisson.pmf(x, mu=10),
        "x_values": np.arange(0, 10)
    },
    "Uniform": {
        "generator": lambda size: uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=size),
        "density": lambda x: uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        "x_values": np.linspace(-2, 2, 1000)
    }
}

# generate and plot histograms
for name, dist in DISTRIBUTIONS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Distribution: {name}")

    for i, size in enumerate(sample_sizes):
        samples = dist["generator"](size=size)
        x_vals = dist["x_values"]
        density_vals = dist["density"](x_vals)

        axes[i].hist(samples, bins='auto', density=True, alpha=0.6, color='pink', label='Histogram')
        axes[i].plot(x_vals, density_vals, 'r-', label='PDF')

        axes[i].set_title(f"Sample size: {size}")
        axes[i].legend(loc='best', frameon=False)
        axes[i].set_xlim([x_vals[0], x_vals[-1]])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    plt.show()

# statistics
repeats = 1000
results = []


def compute_statistics(samples):
    return np.mean(samples), np.median(samples), (np.quantile(samples, 0.25) + np.quantile(samples, 0.75)) / 2


for name, dist in DISTRIBUTIONS.items():
    for size in sample_sizes:
        means, medians, z_qs = [], [], []

        for _ in range(repeats):
            samples = dist["generator"](size=size)
            mean, median, z_q = compute_statistics(samples)
            means.append(mean)
            medians.append(median)
            z_qs.append(z_q)

        results.append([
            name, size, np.mean(means), np.mean(medians), np.mean(z_qs),
            np.mean(np.square(means)), np.mean(np.square(medians)), np.mean(np.square(z_qs)), np.var(means)
        ])


df = pd.DataFrame(results, columns=[
    "Distribution", "Sample Size", "Mean", "Median", "z_q",
    "Mean²", "Median²", "z_q²", "Variance"
])
print(df.to_string(index=False))
