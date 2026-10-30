"""Cluster scatter plot and candidate-K diagnostics."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def chart(title, subtitle):
    figure = Figure(figsize=(10, 6), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    figure.subplots_adjust(left=0.12, right=0.95, bottom=0.22, top=0.78)
    figure.text(0.1, 0.94, title, fontsize=19, weight="bold", color="#15283F")
    figure.text(0.1, 0.88, subtitle, fontsize=10, color="#526275")
    figure.text(0.1, 0.045, "DATA SCIENCE PJ10 | Synthetic mall customer data - exploratory clustering", fontsize=9, color="#526275")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(color="#E5EAF0")
    axes.set_axisbelow(True)
    return figure, axes


def save_plots(assigned, centroids, evaluations, selected, output):
    figure, axes = chart("Mall customer clusters", f"K={selected}; fitted on standardized income and spending score; stars mark centroids")
    colors = ["#2457A7", "#138A87", "#D56E25", "#8D56A5", "#BD4D63", "#6D7C33", "#697787", "#228FB0"]
    for row in centroids.itertuples():
        customers = assigned[assigned.cluster_id == row.cluster_id]
        color = colors[row.cluster_id]
        axes.scatter(customers.annual_income_k, customers.spending_score, color=color, s=35, alpha=0.8, label=f"Cluster {row.cluster_id}")
        axes.scatter(row.annual_income_k, row.spending_score, marker="*", color=color, edgecolors="#15283F", s=260, linewidth=1)
    axes.set(xlabel="Annual income (thousands of synthetic USD)", ylabel="Spending score (1-100)", ylim=(0, 105))
    figure.subplots_adjust(right=0.79)
    axes.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.savefig(output / "clusters.png", dpi=160)
    figure.clear()

    figure, axes = chart("K-means inertia by cluster count", "Elbow diagnostic in standardized space; lower inertia alone does not select the best K")
    axes.plot([row["k"] for row in evaluations], [row["inertia"] for row in evaluations], marker="o", color="#2457A7", linewidth=2)
    axes.axvline(selected, color="#138A87", linestyle="--", label=f"Selected K={selected}")
    axes.set_xticks(range(1, 9))
    axes.set(xlabel="Number of clusters K", ylabel="Within-cluster squared distance (inertia)")
    axes.legend(frameon=False)
    figure.savefig(output / "elbow.png", dpi=160)
    figure.clear()

    figure, axes = chart("Silhouette score by cluster count", "Internal separation diagnostic on fitted data; higher is better")
    valid = evaluations[1:]
    axes.plot([row["k"] for row in valid], [row["silhouette"] for row in valid], marker="o", color="#2457A7", linewidth=2)
    axes.axvline(selected, color="#138A87", linestyle="--", label=f"Selected K={selected}")
    axes.set_xticks(range(2, 9))
    axes.set(xlabel="Number of clusters K", ylabel="Mean silhouette score", ylim=(-1, 1))
    axes.legend(frameon=False)
    figure.savefig(output / "silhouette.png", dpi=160)
    figure.clear()
