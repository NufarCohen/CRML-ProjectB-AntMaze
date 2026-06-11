import copy
import xml.etree.ElementTree as ET



def _rename_names(elem, suffix):
    """
    Rename all named objects inside a copied MuJoCo subtree.
    This prevents duplicate names for body / geom / joint / camera / light.
    """
    for tag in ["body", "geom", "joint", "camera", "light", "site"]:
        for obj in elem.findall(f".//{tag}"):
            name = obj.get("name")
            if name is not None:
                obj.set("name", f"{name}_{suffix}")


def _rename_joint_refs(elem, suffix):
    """
    Actuator motor joints point by name to joints.
    If we copy actuators, their joint refs must also be renamed.
    """
    for obj in elem.iter():
        joint_name = obj.get("joint")
        if joint_name is not None:
            obj.set("joint", f"{joint_name}_{suffix}")


def make_two_ant_tree(tree, ant2_pos="4 4 0.75"):
    """
    Starting from the original OGBench ant.xml tree:
    - keep the original ant as ant1
    - duplicate the ant body as ant2
    - duplicate the actuators for ant2
    Result:
    - model.nu should become 16
    - action[:8] controls original ant
    - action[8:16] controls copied ant
    """
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

    ant2 = copy.deepcopy(ant1)
    ant2.set("name", "torso_ant2")
    ant2.set("pos", ant2_pos)

    _rename_names(ant2, "ant2")

    # Avoid having another tracking light attached to the copied torso.
    for light in list(ant2.findall(".//light")):
        parent = ant2
        # Simple safe removal only if light is direct child somewhere is hard with ElementTree.
        # Leaving renamed lights is usually okay, but duplicated tracking lights can be annoying visually.
        # We keep them renamed for now.

    worldbody.append(ant2)

    # Copy all motors for ant2.
    original_actuators = list(actuator)
    for motor in original_actuators:
        motor2 = copy.deepcopy(motor)
        name = motor2.get("name")
        if name is not None:
            motor2.set("name", f"{name}_ant2")
        _rename_joint_refs(motor2, "ant2")
        actuator.append(motor2)

    return tree