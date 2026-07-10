from ogbench.locomaze.multi_ant_utils import make_multi_ant_tree


def make_two_ant_tree(tree, ant2_pos="4 4 0.75"):
    """Backward-compatible wrapper around make_multi_ant_tree."""
    return make_multi_ant_tree(tree, num_ants=2, ant_positions=[ant2_pos])
