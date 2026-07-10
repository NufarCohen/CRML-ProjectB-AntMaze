import copy
import xml.etree.ElementTree as ET


def _rename_names(elem, suffix):
    """Rename named objects inside a copied MuJoCo subtree."""
    for tag in ["body", "geom", "joint", "camera", "light", "site"]:
        for obj in elem.findall(f".//{tag}"):
            name = obj.get("name")
            if name is not None:
                obj.set("name", f"{name}_{suffix}")


def _rename_joint_refs(elem, suffix):
    """Rename joint refs inside copied actuators."""
    for obj in elem.iter():
        joint_name = obj.get("joint")
        if joint_name is not None:
            obj.set("joint", f"{joint_name}_{suffix}")


_DEFAULT_ANT_POSITIONS = {
    2: ["4 4 0.75"],
    3: ["4 4 0.75", "4 20 0.75"],
    4: ["4 4 0.75", "4 20 0.75", "20 4 0.75"],
}


def make_multi_ant_tree(tree, num_ants=2, ant_positions=None):
    """
    Duplicate the original ant body/actuators to create a multi-ant scene.

    The original ant remains ant1. Additional ants are appended to worldbody
    with duplicated actuators in the same order.
    """
    if num_ants < 1:
        raise ValueError(f"num_ants must be >= 1, got {num_ants}")

    if num_ants == 1:
        return tree

    root = tree.getroot()
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")

    if worldbody is None:
        raise RuntimeError("No <worldbody> found in XML.")
    if actuator is None:
        raise RuntimeError("No <actuator> found in XML.")

    ant1 = worldbody.find("body")
    if ant1 is None:
        raise RuntimeError("No ant <body> found in <worldbody>.")

    if ant_positions is None:
        if num_ants not in _DEFAULT_ANT_POSITIONS:
            raise NotImplementedError(
                f"No default ant positions for num_ants={num_ants}. "
                "Pass ant_positions explicitly."
            )
        ant_positions = _DEFAULT_ANT_POSITIONS[num_ants]

    if len(ant_positions) != num_ants - 1:
        raise ValueError(
            f"Expected {num_ants - 1} extra ant positions, got {len(ant_positions)}"
        )

    original_actuators = list(actuator)

    for ant_idx, pos in enumerate(ant_positions, start=2):
        suffix = f"ant{ant_idx}"
        new_ant = copy.deepcopy(ant1)
        new_ant.set("name", f"torso_{suffix}")
        new_ant.set("pos", pos)
        _rename_names(new_ant, suffix)
        worldbody.append(new_ant)

        for motor in original_actuators:
            motor_copy = copy.deepcopy(motor)
            name = motor_copy.get("name")
            if name is not None:
                motor_copy.set("name", f"{name}_{suffix}")
            _rename_joint_refs(motor_copy, suffix)
            actuator.append(motor_copy)

    return tree
