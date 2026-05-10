"""
LiDAR ground level estimation for the two provided point-cloud datasets.

The point-cloud data is stored as a 2D NumPy array where each row contains
three values: x, y and z. The z-values are used to estimate the ground level
with a histogram-based method.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_FILES = [
    DATA_DIR / "dataset1.npy",
    DATA_DIR / "dataset2.npy",
]

IMAGE_DIR = BASE_DIR / "images"
BIN_WIDTH = 0.1
SEARCH_HEIGHT_ABOVE_PEAK = 1.0


def show_cloud(points_plt):
    """Display a 3D point cloud for manual inspection when needed."""
    ax = plt.axes(projection="3d")
    ax.scatter(points_plt[:, 0], points_plt[:, 1], points_plt[:, 2], s=0.01)
    plt.show()


def show_scatter(x, y):
    """Display a simple 2D scatter plot for manual inspection when needed."""
    plt.scatter(x, y)
    plt.show()


def load_point_cloud(file_path):
    """Load one of the provided NumPy point-cloud files."""
    return np.load(file_path)


def build_z_histogram(point_cloud, bin_width=BIN_WIDTH):
    """Build a histogram from the z-values in the point cloud."""
    z_values = point_cloud[:, 2]
    min_z = np.floor(z_values.min())
    max_z = np.ceil(z_values.max())
    bin_edges = np.arange(min_z, max_z + bin_width, bin_width)
    counts, edges = np.histogram(z_values, bins=bin_edges)
    return counts, edges


def get_ground_level(point_cloud, bin_width=BIN_WIDTH):
    """
    Estimate the ground level using the z-coordinate histogram.

    The largest histogram bin is treated as the main ground peak. The selected
    ground level is the first clear valley after that peak, which works as a
    cutoff for separating the ground plane from the points above it.
    """
    counts, edges = build_z_histogram(point_cloud, bin_width)

    ground_peak_index = int(np.argmax(counts))
    search_start_index = ground_peak_index + 1
    search_end_value = edges[ground_peak_index] + SEARCH_HEIGHT_ABOVE_PEAK
    search_end_index = np.searchsorted(edges, search_end_value, side="right") - 1
    search_end_index = min(search_end_index, len(counts))

    if search_start_index >= search_end_index:
        ground_level = (edges[ground_peak_index] + edges[ground_peak_index + 1]) / 2
        return round(float(ground_level), 2)

    valley_index = search_start_index + int(
        np.argmin(counts[search_start_index:search_end_index])
    )

    ground_level = edges[valley_index + 1]
    return round(float(ground_level), 2)


def save_ground_histogram(point_cloud, dataset_name, ground_level, output_path):
    """
    Save a histogram plot that shows the z-value distribution and selected
    ground level.
    """
    counts, edges = build_z_histogram(point_cloud)
    z_values = point_cloud[:, 2]
    ground_peak_index = int(np.argmax(counts))
    ground_peak_center = (edges[ground_peak_index] + edges[ground_peak_index + 1]) / 2

    plt.figure(figsize=(10, 6))
    plt.hist(z_values, bins=edges)
    plt.axvline(
        ground_peak_center,
        linestyle=":",
        linewidth=2,
        label=f"Ground peak: {ground_peak_center:.2f}",
    )
    plt.axvline(
        ground_level,
        linestyle="--",
        linewidth=2,
        label=f"Selected ground level: {ground_level:.2f}",
    )
    plt.title(f"Z-value histogram for {dataset_name}")
    plt.xlabel("Z value")
    plt.ylabel("Number of points")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def process_dataset(dataset_path):
    """Load a dataset, estimate its ground level, and save its histogram plot."""
    point_cloud = load_point_cloud(dataset_path)
    ground_level = get_ground_level(point_cloud)

    IMAGE_DIR.mkdir(exist_ok=True)
    histogram_path = IMAGE_DIR / f"{dataset_path.stem}_histogram.png"
    save_ground_histogram(
        point_cloud=point_cloud,
        dataset_name=dataset_path.stem,
        ground_level=ground_level,
        output_path=histogram_path,
    )

    points_above_ground = point_cloud[point_cloud[:, 2] > ground_level]

    return {
        "dataset": dataset_path.name,
        "ground_level": ground_level,
        "total_points": point_cloud.shape[0],
        "points_above_ground": points_above_ground.shape[0],
        "histogram": histogram_path,
    }


def main():
    """Run the ground level estimation for both datasets."""
    results = []

    for dataset_path in DATASET_FILES:
        result = process_dataset(dataset_path)
        results.append(result)

    print("Dataset results")
    print("---------------")
    for result in results:
        print(
            f"{result['dataset']}: ground level = {result['ground_level']:.2f}, "
            f"points above ground = {result['points_above_ground']} / "
            f"{result['total_points']}, histogram = {result['histogram']}"
        )


if __name__ == "__main__":
    main()
