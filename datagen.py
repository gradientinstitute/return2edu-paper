"""
Code for simulating data to test inference pipelines.
"""

import networkx as nx
import numpy as np
from econml.orf import DMLOrthoForest, DROrthoForest


def generate_coef(n_features, n_significant):
    """Return an array of coefficients with alternating signs and exponentially decreasing magnitude."""
    n_significant = min(n_features, n_significant)
    indx = np.arange(n_features)
    coef = (-1)**indx*np.exp(-indx/n_significant)
    return coef


class DGPGraph():
    """
    A high level Interface for building Bayesian network data generating processes.

    //TODO remember and document how to use this properly!
    // i.e, how can the dimensionality of a variable be increased (or do I have to create many?)

    Example
    --------

    """
    # TODO add some tracking to keep track of shape of variables & warn if problems arise

    def __init__(self):
        """Construct a new DGP."""
        self.nodes = {}
        self.parents = {}
        self.graph = nx.DiGraph()

    def add_node(self, name, sample_func, parents=None, standardise=False):
        """
        Add a node to the Bayesian Network.

        Parameters
        ----------
        name: str
            The name of the node

        sample_func: function(*tensors) -> pyro.dist(*tensors)
            The sampling function for pyro.sample

        parents: (optional) [str]
            A list of the parents of this node. If None, node must be a root node.

        standardise: (optional) bool
            Should the value of this node be automatically scaled & centered. Default False.

        """
        self.nodes[name] = (sample_func, standardise)
        self.graph.add_node(name)
        if parents is not None:
            self.parents[name] = parents
            for p in parents:
                self.graph.add_edge(p, name)

        self.shapes = self._check_func_returns()

    def get_function(self, node):
        """Return the function for generating data for a node given its parents."""
        return self.nodes[node][0]

    def get_parents(self, node):
        "Return the list of parents for the given node or an empty list if there are none."
        if node in self.parents:
            return self.parents[node]
        return []

    def _check_func_returns(self):
        """Checks and store the shape of the results returned by sample."""
        n_test = 3
        shapes = {}
        values = self.sample(n_test)

        for node, data in values.items():
            t = type(data)
            if t != np.ndarray:
                raise ValueError(
                    f'Invalid sample function for {node}. Should return a np.ndarray returned type {t}'
                )

            s = data.shape

            msg = f"Invalid shape {s} from sample({node},n={n_test}). Result for {node} must be np.array(n,) or (n, .)"

            if len(s) not in [1, 2]:
                raise ValueError(msg)
            else:
                if s[0] != n_test:
                    raise(ValueError(msg))

            shapes[node] = s

        return shapes

    def _expand(self, node, value, n):
        """Explicitly broadcast interventional values to the correct shape."""
        s = self.shapes[node]
        if len(s) == 1:
            v = np.ones(n, dtype=float)
        elif len(s) == 2:
            v = np.ones((n, s[1]), dtype=float)
        else:
            raise ValueError(f"Shape for {node} must be one or dimensional but was {s}")

        return v*value

    def draw_graph(self):
        """Draw the DAG for the data generating process."""
        nx.draw(self.graph, with_labels=True)

    def sample(self, n, interventions=None):
        """
        Sample values from each node following the generative process.

        Parameters
        ----------
        n: int
            The number of samples to draw

        interventions: {str:value}
            A dict from a variable name to a valid value for that variable.

        Returns:
        values: {str:np.array}
            A dict from variable name to sampled values.
        """
        values = {}
        for node in nx.topological_sort(self.graph):
            if interventions is not None and node in interventions:
                value = self._expand(node, interventions[node], n)
                values[node] = value
            else:
                parents = self.get_parents(node)
                values_parent = [values[p] for p in parents]
                func, standardise = self.nodes[node]
                value = func(*values_parent, n=n)
                if standardise:
                    pass  # TODO implement
                values[node] = value

        return values

    def ate(self, n, treatment_node, outcome_node, treatment_val=1, control_val=0):
        """Compute the estimated Average Treatment Effect based on a sample of size n."""
        s1 = self.sample(n, interventions={treatment_node: treatment_val})
        s0 = self.sample(n, interventions={treatment_node: control_val})
        ate = s1[outcome_node].mean() - s0[outcome_node].mean()
        # TODO add standard error in ate
        return ate

    def cate(self, n, treatment_node, outcome_node, condition_node, condition_values, treatment_val=1, control_val=0):
        condition_shape = self.shapes[condition_node]
        if len(condition_shape) > 1:
            raise NotImplementedError(
                "CATE estimation not currently supported for variables with dimensionality > 1."
            )
        if len(self.get_parents(condition_node)) > 0:
            raise NotImplementedError(
                "CATE estimation not currently supported for non root nodes in the DGP."
            )

        result = np.zeros(len(condition_values))
        for i, v in enumerate(condition_values):
            s1 = self.sample(n, interventions={condition_node: v, treatment_node: treatment_val})
            s0 = self.sample(n, interventions={condition_node: v, treatment_node: control_val})
            cate = s1[outcome_node].mean() - s0[outcome_node].mean()
            result[i] = cate
        return result

    def fix_standardisation_parameters(self, n):
        return self
