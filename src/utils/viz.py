# LOAD MODULES
# Standard library
from typing import List, Optional, Any, Callable

# Third party
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use('science')
from matplotlib import colormaps as cm


# CUSTOM FUNCTIONS
def normalize_variables(x: np.ndarray) -> np.ndarray:
    """
    Normalizes the variables in the given covariates matrix.

    The function normalizes each variable (column) in the covariates matrix to the range [0, 1] using min-max normalization.

    Parameters:
        x (np.ndarray): The covariates matrix, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.

    Returns:
        np.ndarray: The normalized covariates matrix.
    """
    num_covariates = x.shape[1]
    for idx in range(num_covariates):
        x[:, idx] = (x[:, idx] - np.min(x[:, idx])) / (np.max(x[:, idx]) - np.min(x[:, idx]))

    return x


class Visualizer():
    """
    The Visualizer class is used to create visualizations of the data.
    It takes in covariates and doses as inputs and provides methods for creating various types of plots.
    The class also supports normalization of covariates and doses, and allows for stratification of doses.

    Attributes:
        x (Any): The covariates to be visualized.
        d (Any): The doses to be visualized.
        t (Any): The treatment to be visualized.
        num_dose_strata (int): The number of dose strata. Default is 10.
        min_obs_per_strata (Optional[int]): The minimum number of observations per strata. If None, all observations are considered. Default is None.
        normalize_covariates (bool): If True, the covariates are normalized. If False, the covariates are not normalized. Default is True.
        normalize_doses (bool): If True, the doses are normalized. If False, the doses are not normalized. Default is True.
    """

    def __init__(
            self,
            x: Any,
            d: Any,
            t: Any = None,
            num_dose_strata: int = 10,
            min_obs_per_strata: Optional[int] = None,
            normalize_covariates: bool = True,
            normalize_doses: bool = True,
    ) -> None:
        """
        Initializes a new instance of the Visualizer class.

        Parameters:
            x (Any): The covariates to be visualized.
            d (Any): The doses to be visualized.
            t (Any): The treatment to be visualized.
            num_dose_strata (int, optional): The number of dose strata. Defaults to 10.
            min_obs_per_strata (Optional[int], optional): The minimum number of observations per strata. Defaults to None.
            normalize_covariates (bool, optional): If True, the covariates are normalized. Defaults to True.
            normalize_doses (bool, optional): If True, the doses are normalized. Defaults to True.
        """
        self.x = x
        self.d = d
        # Save treatments, or set dummies
        if t is not None:
            self.t = t
        else:
            self.t = np.zeros(self.x.shape[0])
        self.num_dose_strata = num_dose_strata
        self.min_obs_per_strata = min_obs_per_strata
        self.normalize_covariates = normalize_covariates
        self.normalize_doses = normalize_doses
        self.initialized = False

    def init(
            self,
    ):
        """
        This function initializes the Visualizer class and has to be run before other functions.
        """
        print("Initializing...")

        # Main part
        # Save info
        num_obs = self.x.shape[0]
        num_cov = self.x.shape[1]
        strata = np.linspace(0, 1, self.num_dose_strata + 1)
        # Add epsilon at boundaries
        strata[0] = strata[0] - np.finfo(float).eps
        strata[-1] = strata[-1] + np.finfo(float).eps

        # Normalize
        if self.normalize_covariates:
            self.x = normalize_variables(self.x)

        if self.normalize_doses:
            self.d = (self.d - np.min(self.d)) / (np.max(self.d) - np.min(self.d))

        # Get pairwise distances
        distance_matrix = squareform(pdist(self.x))

        # Normalize distances
        max_val = np.max(distance_matrix)
        min_val = np.partition(np.ravel(distance_matrix), num_obs)[num_obs]
        distance_matrix = (distance_matrix - min_val) / (max_val - min_val)

        # Transform to doses to strata
        obs_strata = np.digitize(self.d, strata) - 1

        # Generate counting matrix
        strata_counts_per_obs = []

        # Iterate over observations
        for obs in tqdm(range(num_obs), desc="Iterate over observations", leave=False):
            # Save strata counter for obs
            helper_strata_counter = [[] for n in range(self.num_dose_strata)]
            # Save distances to obs
            distances_to_obs = distance_matrix[obs, :]
            idx_sorted_by_dist = distances_to_obs.argsort()

            # Iterate over neighbours
            for ranking, neighbor_idx in enumerate(idx_sorted_by_dist[1:]):
                neighbor_strata = obs_strata[neighbor_idx]
                helper_strata_counter[neighbor_strata].append(ranking)
                # Check if minimum number of neighbours per strata is reached
                if self.min_obs_per_strata is not None:
                    if min([len(ls) for ls in helper_strata_counter]) >= self.min_obs_per_strata:
                        break

            # Append to strata_counts_per_obs
            strata_counts_per_obs.append(helper_strata_counter)

        # Save in class object
        self.num_obs = num_obs
        self.num_cov = num_cov
        self.strata = strata
        self.obs_strata = obs_strata
        self.distance_matrix = distance_matrix
        self.strata_counts_per_obs = strata_counts_per_obs

        # Set init flag to True
        self.initialized = True

    def tsne_plot(
            self,
            num_hist_bins: int = 50,
            sub_pop_indicator: Optional[List[str]] = None,
    ) -> None:
        """
        This function plots the t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization.
        The 'sub_pop_indicator' parameter determines the sub population indicator.

        Parameters:
        num_hist_bins (int): The number of bins for the histogram. Default is 50.
        sub_pop_indicator (List[str]): The sub population indicator. If None, all observations are considered. Default is None.

        Returns:
        None
        """
        if self.initialized is False:
            self.init()

        # Create dummy sub_pop_indicator if None
        if sub_pop_indicator is None:
            sub_pop_indicator = np.repeat(0, self.num_obs)
        # Do PCA if number of covariates is high
        if self.num_cov >= 10:
            pca = PCA(n_components=10)
            components = pca.fit_transform(self.x)
        else:
            components = self.x

        # Do tSNE
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(components)

        # Plot
        fig, axes = plt.subplot_mosaic("AAAB;AAAB;AAAB")
        fig.suptitle("t-SNE plot")
        fig.set_size_inches(5, 4)

        # Set color
        cm = sns.color_palette("ch:d=.25,l=.75", num_hist_bins)

        # Scatter
        sns.scatterplot(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=self.d,
            ax=axes["A"],
            style=sub_pop_indicator
        )
        axes["A"].get_legend().set_visible(False)

        # Set Labels
        axes["A"].set_xlabel("Dimension 1")
        axes["A"].set_ylabel("Dimension 2")

        hist = sns.histplot(y=self.d, bins=num_hist_bins, stat="density", ax=axes["B"])
        for bin_, i in zip(hist.patches, cm):
            bin_.set_facecolor(i)

        axes["B"].set_xlabel("Density")
        axes["B"].set_ylabel("Dose")
        axes["B"].set_ylim((0, 1))
        axes["B"].yaxis.tick_right()
        axes["B"].yaxis.set_label_position("right")
        plt.show()

    def density_plot(
            self,
            distance: bool = False,
            per_strata: bool = False,
            num_obs_per_strata: int = 100,
            sub_pop_indicator: Optional[List[str]] = None,
    ) -> None:
        """
        This function plots the density. It can plot per strata based on the 'per_strata' parameter.
        The 'sub_pop_indicator' parameter determines the sub population indicator.

        Parameters:
            distance (bool): If True, the values plotted are distances. If False, the values plotted are percentiles. Default is False.
            per_strata (bool): If True, a separate plot is created for each strata. If False, a single plot is created. Default is False.
            num_obs_per_strata (int): The number of observations per strata to plot. Default is 100.
            sub_pop_indicator (List[str]): The sub population indicator. If None, all observations are considered. Default is None.

        Returns:
            None
        """
        if self.initialized is False:
            self.init()

        plotting_values = []
        plotting_strata = []
        plotting_group = []

        # Create dummy sub_pop_indicator if None
        if sub_pop_indicator is None:
            sub_pop_indicator = np.repeat("All observations", self.num_obs)

        # Define color map
        cm = sns.color_palette("ch:d=.25,l=.75", 2)

        # Iterate over obs
        for obs in tqdm(range(self.num_obs), desc="Iterate over observations", leave=False):
            values_per_strata = []
            # Iterate over stratas
            for strata in range(self.num_dose_strata):
                value = (self.strata_counts_per_obs[obs][strata][num_obs_per_strata])
                # Return distance or percentile
                if distance:
                    value = self.distance_matrix[obs, value]
                else:
                    value = value / (self.num_obs - 1)

                # Append to array
                values_per_strata.append(value)
            if per_strata:
                # Append
                plotting_values.extend(values_per_strata)
                plotting_strata.extend(range(self.num_dose_strata))
                plotting_group.extend([sub_pop_indicator[obs] for i in range(self.num_dose_strata)])
            else:
                # Append max to nn_percentages
                plotting_values.append(max(values_per_strata))
                plotting_group.append(sub_pop_indicator[obs])

        # Save as np array
        plotting_values = np.array(plotting_values)
        plotting_group = np.array(plotting_group)
        if per_strata:
            plotting_strata = np.array(plotting_strata).astype(str)

        # Plot
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Density plot: " + ("NN distance" if distance else "NN percentile"))
        fig.set_size_inches(8, 4)

        # Plot density
        if per_strata:
            sns.violinplot(x=plotting_strata, y=plotting_values, split=True, hue=plotting_group, ax=ax, inner="quart",
                           density_norm="area", palette=cm)
            _ = np.round(np.linspace(0, 1, self.num_dose_strata + 1), 2).astype(str)
            xlabs = [("(" + _[i] + "," + _[i + 1] + "]") for i in range(self.num_dose_strata)]
            ax.set_xticks(range(self.num_dose_strata), labels=xlabs)
            plt.ylim((0, 1))
            plt.xlabel("Dose strata")
            plt.ylabel(("Distance required" if distance else "NN percentile"))
        else:
            sns.violinplot(x=plotting_values, hue=plotting_group, split=True, ax=ax, inner="quart", density_norm="area",
                           palette=cm)
            plt.xlim((0, 1))
            plt.xlabel(
                ("Distance required for dose coverage" if distance else "NN percentile required for dose coverage"))
            plt.ylabel("Density")
        plt.show()

    def nn_distance(
            self,
            sample_size: float = 1.,
            plot_ever_x_neighbour: int = 50,
            x_on_log_scale: bool = True,
            sub_pop_indicator: Optional[List[str]] = None,
    ) -> None:
        """
        This function plots the n-nearest neighbor distance. It can plot on a log scale based on the
        'x_on_log_scale' parameter. The 'sub_pop_indicator' parameter determines the sub population indicator.

        Parameters:
            sample_size (float): The size of the sample to plot. Default is 1.
            plot_ever_x_neighbour (int): The number of neighbors to plot. Default is 50.
            x_on_log_scale (bool): If True, the x-axis is on a log scale. If False, the x-axis is on a linear scale. Default is True.
            sub_pop_indicator (List[str]): The sub population indicator. If None, all observations are considered. Default is None.

        Returns:
            None
        """
        if self.initialized is False:
            self.init()

        distances = []
        neighbor_idx = []
        sub_pop = []

        # Create dummy sub_pop_indicator if None
        if sub_pop_indicator is None:
            sub_pop_indicator = np.repeat("All observations", self.num_obs)

        # Define color map
        cm = sns.color_palette("ch:d=.25,l=.75", 2)

        num_obs = int(self.num_obs * sample_size)
        sample_ids = np.random.permutation(self.num_obs)[:num_obs]

        # Iterate over samples
        for obs in tqdm(sample_ids, desc="Iterate over observations"):
            # Get distances to observation
            distances_to_obs = self.distance_matrix[obs, :]
            idx_sorted_by_dist = distances_to_obs.argsort()
            # Iterate over neighbors
            for idx in range(1, self.num_obs, plot_ever_x_neighbour):
                # Append distances
                distances.append(distances_to_obs[idx_sorted_by_dist[idx]])
                # Append neirbor index
                neighbor_idx.append(idx)
                # Append sub_pop
                sub_pop.append(sub_pop_indicator[obs])

        fig, ax = plt.subplots(1, 1)
        fig.suptitle("n-Nearest neighbor distance")
        fig.set_size_inches(8, 4)
        ax.set_ylim((0, 1))
        if x_on_log_scale:
            ax.set_xscale("log")
        ax.set_ylabel("Distance")
        ax.set_xlabel("n-Nearest neighbor")

        ax = sns.lineplot(x=neighbor_idx, y=distances, errorbar="sd", hue=sub_pop, palette=cm)
        plt.show()

    def coverage_per_observation(
            self,
            obs_id: int = 1,
            distance: bool = False,
            violin_plot: bool = True,
    ) -> None:
        """
        This function plots the coverage per observation. It can either plot a violin plot or a box plot based on the
        'violin_plot' parameter. The 'distance' parameter determines whether the values plotted are distances or percentiles.

        Parameters:
        obs_id (int): The id of the observation to plot. Default is 1.
        distance (bool): If True, the values plotted are distances. If False, the values plotted are percentiles. Default is False.
        violin_plot (bool): If True, a violin plot is plotted. If False, a box plot is plotted. Default is True.

        Returns:
        None
        """
        if self.initialized is False:
            self.init()

        # Plotting arrays
        plotting_strata = []
        plotting_values = []
        plotting_color = []  # Dummy

        # Define color map
        cm = sns.color_palette("ch:d=.25,l=.75", 2)

        # Iterate over strata
        for s in range(self.num_dose_strata):
            values = self.strata_counts_per_obs[obs_id][s]
            if distance:
                values = [self.distance_matrix[obs_id, i] for i in values]
            else:
                values = [(i / (self.num_obs - 1)) for i in values]

            # Append
            plotting_strata.extend([s] * len(values))
            plotting_values.extend(values)
            plotting_color.extend(["Neighbors"] * len(values))

        fig, ax = plt.subplots(1, 1)
        fig.suptitle("n-Nearest neighbor distance")
        fig.set_size_inches(8, 4)

        plt.xlabel("Dose strata")
        plt.ylabel(("Distance required" if distance else "NN percentile"))

        if violin_plot:
            plotter = sns.violinplot
        else:
            plotter = sns.boxplot
        plotter(x=plotting_strata, y=plotting_values, ax=ax, hue=plotting_color, palette=cm, width=0.5)
        _ = np.round(np.linspace(0, 1, self.num_dose_strata + 1), 2).astype(str)
        xlabs = [("(" + _[i] + "," + _[i + 1] + "]") for i in range(self.num_dose_strata)]
        ax.set_xticks(range(self.num_dose_strata), labels=xlabs)
        plt.show()

    def dr_curves(
            self,
            ground_truth: Callable,
            n_curves: int = 1000,
    ):
        """
        For any synthetic dataset, this function plots the dose-response curves for a given number of observations and color-codes based on the factual dose.
        Requires a ground truth function that calculates the outcome variable for a given dataset and doses.
        """
        # Settings
        alpha_line = 0.1
        alpha_dot = 0.25
        alpha_tsne = 0.67
        size_tsne = 15
        col_map = "viridis"
        # Plotting points
        plotting_samples = np.linspace(0.001, 0.999, 65)

        # Bind n_curves
        if n_curves > self.x.shape[0]:
            n_curves = self.x.shape[0]

        # Sample observations
        sample_ids = np.random.choice(self.x.shape[0], n_curves, replace=False)
        samples = self.x[sample_ids]
        sample_ds = self.d[sample_ids]
        samples_ts = self.t[sample_ids]

        # Ini plot
        fig, axes = plt.subplot_mosaic("AAACCC;AAACCC;BBBCCC")
        fig.suptitle("Dose-response curves")
        # fig.tight_layout()
        fig.set_size_inches(6, 4)

        # Generate dr curves
        axes["A"].set_title("Dose-response curves and factual doses")
        # Plot dr curves
        for obs_id in sample_ids:
            axes["A"].plot(
                plotting_samples,
                [ground_truth(samples[obs_id].reshape(1, -1), d, samples_ts[obs_id]).item() for d in plotting_samples],
                alpha=alpha_line,
                color=cm[col_map](sample_ds[obs_id])
            )

        # Plot factual outcomes
        for obs_id in sample_ids:
            axes["A"].plot(
                sample_ds[obs_id],
                ground_truth(samples[obs_id].reshape(1, -1), sample_ds[obs_id], samples_ts[obs_id]).item(),
                'o',
                color='black',
                markersize=1,
                alpha=alpha_dot)

        # Plot histogram of factual doses
        axes["B"].set_title("Dose distribution")
        hist, hist_edges = np.histogram(sample_ds, 65, density=True, range=(0, 1))
        axes["B"].bar(hist_edges[:-1] + 0.5 * (hist_edges[1] - hist_edges[0]),
                      hist,
                      color=[cm[col_map](val) for val in hist_edges],
                      width=hist_edges[1] - hist_edges[0])

        # Add tSNE plot
        axes["C"].set_title("t-SNE plot")
        tsne = TSNE(2, random_state=42, perplexity=30, verbose=2, n_iter=1000)
        tsna_data = tsne.fit_transform(samples)

        axes["C"].scatter(
            x=tsna_data[:, 0],
            y=tsna_data[:, 1],
            alpha=alpha_tsne,
            c=cm[col_map](sample_ds),
            s=size_tsne,
            edgecolors='none',
        )

        plt.show()


def tsne_plot(
        x: np.ndarray,
        d: np.ndarray,
        w: float = 4,
        h: float = 4,
        file_name: str = "tsne_plot.pdf",
        labels: bool = True,
) -> None:
    """
    Creates and saves a t-SNE plot of the given data.

    Parameters:
        x (np.ndarray): The input data for t-SNE.
        d (np.ndarray): The dose values for coloring the plot.
        w (float, optional): The width of the plot. Defaults to 4.
        h (float, optional): The height of the plot. Defaults to 4.
        file_name (str, optional): The name of the file to save the plot. Defaults to "tsne_plot.pdf".
        labels (bool, optional): If True, includes axis labels. Defaults to True.
    """
    col_map = "viridis"

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(x)

    fig, axes = plt.subplot_mosaic("A")
    fig.set_size_inches(w, h)

    axes["A"].scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        alpha=1,
        c=cm[col_map](d),
        s=15,
        edgecolors='none',
    )

    if labels is False:
        # Remove the y-axis
        axes["A"].set_yticks([])
        axes["A"].set_xticks([])

    plt.savefig(file_name)


def dose_plot(
        d: np.ndarray,
        w: float = 4,
        h: float = 1,
        samples: int = None,
        file_name: str = "dose_plot.pdf",
        color: str = None,
        num_bins: int = 33,
        labels: bool = False,
) -> None:
    """
    Creates and saves a dose distribution plot.

    Parameters:
        d (np.ndarray): The dose values to plot.
        w (float, optional): The width of the plot. Defaults to 4.
        h (float, optional): The height of the plot. Defaults to 1.
        samples (int, optional): The number of samples to plot. If None, all samples are used. Defaults to None.
        file_name (str, optional): The name of the file to save the plot. Defaults to "dose_plot.pdf".
        color (str, optional): The color of the bars. If None, a colormap is used. Defaults to None.
        num_bins (int, optional): The number of bins for the histogram. Defaults to 33.
        labels (bool, optional): If True, includes axis labels. Defaults to False.
    """
    col_map = "viridis"

    # Set number of samples
    if samples is None:
        samples = len(d)
    else:
        samples = min(samples, len(d))

    # Sample
    ids = np.random.choice(len(d), samples, replace=False)
    d = d[ids]

    # Get histogram
    hist, hist_edges = np.histogram(d, num_bins, density=True, range=(0, 1))

    fig, axes = plt.subplot_mosaic("A")

    if color is None:
        color = [cm[col_map](val) for val in hist_edges]

    axes["A"].bar(
        hist_edges[:-1] + 0.5 * (hist_edges[1] - hist_edges[0]),
        hist,
        color=color,
        width=hist_edges[1] - hist_edges[0])

    if labels is False:
        # Remove the y-axis
        axes["A"].set_yticks([])
    else:
        axes["A"].set_ylabel("Density")
        axes["A"].set_xlabel("Dose")

    # Adjust margins
    fig.set_size_inches(w, h)

    plt.savefig(file_name)


def dr_plot(
        x: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
        gt: Callable,
        w: float,
        h: float,
        samples: int = None,
        file_name: str = "dr_plot.pdf",
) -> None:
    """
    Creates and saves a dose-response plot.

    Parameters:
        x (np.ndarray): The input features.
        d (np.ndarray): The dose values.
        t (np.ndarray): The treatment values.
        gt (Callable): The ground truth function.
        w (float): The width of the plot.
        h (float): The height of the plot.
        samples (int, optional): The number of samples to plot. If None, all samples are used. Defaults to None.
        file_name (str, optional): The name of the file to save the plot. Defaults to "dr_plot.pdf".
    """
    col_map = "viridis"

    # Plotting points
    plotting_samples = np.linspace(0.001, 0.999, 65)

    # Set number of samples
    if samples is None:
        samples = len(d)
    else:
        samples = min(samples, len(d))

    # Sample
    ids = np.random.choice(len(d), samples, replace=False)

    fig, axes = plt.subplot_mosaic("A")

    axes["A"].set_ylabel("Outcome")

    # Plot curves
    for id in ids:
        axes["A"].plot(
            plotting_samples,
            [gt(x[id].reshape(1, -1), help_d, t[id]).item() for help_d in plotting_samples],
            alpha=0.25,
            color=cm[col_map](d[id])
        )

    # Plot factuals
    for id in ids:
        axes["A"].plot(
            d[id],
            gt(x[id].reshape(1, -1), d[id], t[id]).item(),
            'o',
            color='black',
            markersize=1,
            alpha=0.5)

    # Remove the y-axis
    axes["A"].set_xticks([])

    # Adjust margins
    fig.set_size_inches(w, h)

    plt.savefig(file_name)


def dose_dr_plot(
        x: np.ndarray,
        d: np.ndarray,
        t: np.ndarray,
        gt: Callable,
        w: float,
        h: float,
        samples: int = None,
        num_bins: int = 33,
        file_name: str = "ddr_plot.pdf",
) -> None:
    """
    Creates and saves a combined dose-response and dose distribution plot.

    Parameters:
        x (np.ndarray): The input features.
        d (np.ndarray): The dose values.
        t (np.ndarray): The treatment values.
        gt (Callable): The ground truth function.
        w (float): The width of the plot.
        h (float): The height of the plot.
        samples (int, optional): The number of samples to plot. If None, all samples are used. Defaults to None.
        num_bins (int, optional): The number of bins for the histogram. Defaults to 33.
        file_name (str, optional): The name of the file to save the plot. Defaults to "ddr_plot.pdf".
    """
    col_map = "viridis"

    # Plotting points
    plotting_samples = np.linspace(0.001, 0.999, 65)

    # Set number of samples
    if samples is None:
        samples = len(d)
    else:
        samples = min(samples, len(d))

    # Sample
    ids = np.random.choice(len(d), samples, replace=False)

    fig, axes = plt.subplot_mosaic("AAAA;AAAA;BBBB")

    # DR plot

    axes["A"].set_ylabel("Outcome")

    # Plot curves
    for id in ids:
        axes["A"].plot(
            plotting_samples,
            [gt(x[id].reshape(1, -1), help_d, t[id]).item() for help_d in plotting_samples],
            alpha=0.25,
            color=cm[col_map](d[id])
        )

    # Plot factuals
    for id in ids:
        axes["A"].plot(
            d[id],
            gt(x[id].reshape(1, -1), d[id], t[id]).item(),
            'o',
            color='black',
            markersize=1,
            alpha=0.5)

    # Remove the y-axis
    axes["A"].set_xticks([])

    # Dose plot
    # Get histogram
    hist, hist_edges = np.histogram(d, num_bins, density=True, range=(0, 1))
    color = [cm[col_map](val) for val in hist_edges]

    axes["B"].bar(
        hist_edges[:-1] + 0.5 * (hist_edges[1] - hist_edges[0]),
        hist,
        color=color,
        width=hist_edges[1] - hist_edges[0])

    axes["B"].set_ylabel("Density")
    axes["B"].set_xlabel("Dose")

    # Align y-axis labels
    axes["A"].get_yaxis().set_label_coords(-0.1, 0.5)
    axes["B"].get_yaxis().set_label_coords(-0.1, 0.5)

    # Adjust margins
    fig.set_size_inches(w, h)

    plt.savefig(file_name)


def plot_dose_response(
        data,
        split,
        model,
        name,
        plot_settings,
        optimization_settings,
        decision_vars=None
) -> None:
    """
    Plots the dose-response curves for the given data and model.

    Parameters:
        data: The dataset containing the features and outcomes.
        split: The data split to use ('train' or 'test').
        model: The model used to predict the outcomes.
        name: The name of the plot.
        plot_settings: The settings for the plot.
        optimization_settings: The settings for the optimization.
        decision_vars: The decision variables, if any. Defaults to None.
    """
    ntreatments_plot = plot_settings["ntreatments"]
    ntreatments_optimization = optimization_settings["ntreatments_list"][-1]  # Take the last value in the list

    dose_range = np.linspace(0, 1, ntreatments_plot)

    response = data.ground_truth

    if split == "train":  # Checking if split is "train"
        x_data = data.x_train  # Using x_train if split is "train"
        y_data = data.y_train  # Using y_train if split is "train"
        d_data = data.d_train  # Using d_train if split is "train"
        t_data = data.t_train  # Using t_train if split is "train"
    else:
        x_data = data.x_test
        y_data = data.y_test
        d_data = data.d_test
        t_data = data.t_test

    num_obs = x_data.shape[0]  # Using x_data instead of data.x_test

    y_estlist = []
    y_gtlist = []

    residual_factuals = []  # List to store residuals

    for i in range(ntreatments_plot + 1):
        dose = i / ntreatments_plot
        y_est = model.predict(x_data, np.ones(num_obs) * dose, d_data)
        y_estlist.append(y_est)

    for i in range(ntreatments_plot + 1):
        dose = i / ntreatments_plot
        y_gt = response(x_data, np.ones(num_obs) * dose, d_data)
        y_gtlist.append(y_gt)

    if plot_settings["plot_dose_density"]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3), sharex=True, gridspec_kw={'height_ratios': [7.5, 2.5]})
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [7.5, 2.5, 2.5]})
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3), sharex=True)

    if plot_settings["plot_gt_scatter"]:
        ax1.scatter(d_data, response(x_data, d_data, t_data), s=3, alpha=0.5, color='blue', label="True response")
    if plot_settings["plot_est_scatter"]:
        ax1.scatter(d_data, model.predict(x_data, d_data, t_data), s=3, alpha=0.5, color='orange',
                    label="Estimated response")

    if plot_settings["plot_vlines"]:
        for x, y_gt_dot, y_est_dot in zip(d_data, response(x_data, d_data, t_data),
                                          model.predict(x_data, d_data, t_data)):
            ax1.vlines(x, y_gt_dot, y_est_dot, linestyles='dotted', colors='grey', alpha=1)
    #           residual_factuals.append(abs(y_gt_dot - y_est_dot))

    if plot_settings["plot_gt_curve"]:
        for i in range(num_obs):
            ax1.plot(dose_range, [y_gtlist[j][i] for j in range(ntreatments_plot)], linewidth=0.3, linestyle='-',
                     alpha=0.12,
                     color='blue')
        ax1.plot([], [], linestyle='-', markersize=1, alpha=1, color='blue', label="True CADR")

    if plot_settings["plot_est_curve"]:
        for i in range(num_obs):
            ax1.plot(dose_range, [y_estlist[j][i] for j in range(ntreatments_plot)], linewidth=0.3, linestyle='-',
                     alpha=0.24,
                     color='orange')
        ax1.plot([], [], linestyle='-', markersize=1, alpha=1, color='orange', label="Estimated CADR")

    if plot_settings["plot_decision_vars"] and decision_vars is not None:

        A = optimization_settings["protected_attribute"]

        for i in range(num_obs):
            color = 'r' if x_data[i, A] == 1 else 'g'
            for j in range(ntreatments_optimization):
                if decision_vars[i + 1, j + 1].value() == 1:
                    ax1.scatter((1 / (ntreatments_optimization - 1)) * j, y_estlist[j][i], marker='o',
                                s=x_data[i, 0] * 20,
                                alpha=0.7, facecolors='none', edgecolors=color)

        ax1.scatter([], [], marker='o', s=10, alpha=1, facecolors='none', edgecolors='r',
                    label="Decision variable. Prot==1")
        ax1.scatter([], [], marker='o', s=10, alpha=1, facecolors='none', edgecolors='g',
                    label="Decision variable. Prot==0")

    ax1.set_ylabel("$Y$")
    ax1.set_xlabel("$S$")
    ax1.title.set_text(f"CADR: S-Learner (rf)")
 #   ax1.title.set_text(f"CADR: S-Learner (mlp)")
 #   ax1.title.set_text(f"CADR: DRNet")
 #   ax1.title.set_text(f"CADR: VCNet")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)  # Setting y-axis limits to 0 and 1

    if plot_settings["plot_dose_density"]:
        ax2.hist(d_data, bins=20, color='blue', alpha=0.8, label='Observed Dosage Histogram')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Dose')
        ax2.set_ylabel('Frequency')

        plt.title('Observed Dosage Density')
        plt.tight_layout()

    if plot_settings['save_fig']:
        plt.savefig(f'exp1_cadr_{name}.pdf')
        plt.savefig(f'exp1_cadr_{name}.png')
        plt.savefig(f'exp1_cadr_{name}.jpg')
        print("figure saved")

    plt.show()
