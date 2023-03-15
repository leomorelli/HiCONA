"""Module containing functions to generate synthetic files

NOTE: These files are meant for testing purposes only, such as memory 
footprint, since they make arbitrary distribution assumptions which are 
not based on any biological experiment or observation.
"""

from os import path
from random import randint, sample

from pandas import DataFrame
from scipy.stats import binom, poisson


def gen_synthetic_pixels(
	    num_bins: int,
	    p_edges: float,
	    mu_counts: int,
	    save_folder: str = None,
	    rng_seed: int = None):
    """Return a pandas DataFrame, in coo format, of simulated data 

    Given a number of sequential bins, with id from 0 to num_bins-1, for each 
    bin define a random number of interacting bins among those with a higher  
    id, then assign a random number of counts to each pair. The fraction of 
    interacting bins is sampled from a binomial distribution with parameters
    n = number of bins with higher id, p = p_edges. The number of counts is 
    sampled from a Poisson distribution with parameter mu = mu_counts. The 
    return is a pd.DataFrame in coo format (bin1_id, bin2_id, count).

    Keyword arguments:
    num_bins -- (int) Number of sequential bins (named starting from 0)
    p_edges -- (float) Probability parameter of the binomial distribution
    mu_counts -- (int) Mu parameter of the Poisson distribution
    save_folder -- (str) If present, output folder for the coo (default: None)
    rng_seed -- (int) Random number generation seed (default: None)

    Returns:
    coo -- (pd.DataFrame) Pandas dataframe in coo form (bin1, bin2, count)
    """

    # Input validation
    if not isinstance(num_bins, int):
        raise ValueError("Number of bins must be positive integer.")
    if num_bins < 2:
        raise ValueError("Number of bins must be at least two 2.")

    if type(p_edges) not in (int, float):
        raise ValueError("Probability value must be float or integer.")
    if p_edges < 0 or p_edges > 1:
        raise ValueError("Probability value must be in the range [0,1].")

    if type(mu_counts) not in (int, float):
        raise ValueError("Expected counts value must be float of integer.")
    if mu_counts < 0:
        raise ValueError("Expected counts value must be non-negative.")

    if save_folder is not None and not isinstance(save_folder, str):
        raise ValueError("Save folder path must be in string form.")
    if isinstance(save_folder, str) and not path.isdir(save_folder):
        raise ValueError("Save folder path must be an existing directory.")

    if rng_seed is not None and not isinstance(rng_seed, int):
        raise ValueError("Rng seed must be of integer type.")
    if isinstance(rng_seed, int) and rng_seed < 0:
        raise ValueError("Rng seed must be a non-negative integer.")

    # Set RNG seed for reproducibility
    if rng_seed is None:
        rng_seed = randint(0, 100000)
    binom.random_state = rng_seed
    poisson.random_state = rng_seed

    # Generate the pairs bin1-bin2
    coo = [
        [bin1, bin2]
        for bin1 in range(num_bins)
        for bin2 in sorted(sample(
        	range(bin1+1, num_bins),
        	binom.rvs(num_bins-(bin1+1), p_edges))
            )
        ]

    # Generate counts values
    coo = DataFrame(coo, columns=["bin1_id", "bin2_id"])
    coo["count"] = poisson.rvs(mu_counts, size=coo.shape[0])

    # Save to file if required
    p_perc = int(p_edges * 100)  # To avoid dots in output file name
    f_name = f"synt_{num_bins}bins_{p_perc}p_{mu_counts}mu_{rng_seed}rng.csv"
    if save_folder is not None:
        coo.to_csv(save_folder + "/" + f_name, index=False)

    return coo
