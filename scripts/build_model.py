"""Derive the 6-DOF nonredundant FR3 MJCF used throughout this implementation.

The vendored Menagerie ``franka_fr3/fr3.xml`` has 7 joints. Sections III-V of
Khatib (1987) derive the end-effector equations of motion under the assumption
that the manipulator is *not* redundant (m0 = n, eq. 9-10). Redundancy
resolution (dynamically-consistent pseudoinverse, null-space control) is
Sec. VI-VII of the paper and is out of scope for this implementation.

To make the OSC law in ``controller.py`` a complete, well-posed solution
exactly as the paper derives it (Gamma = J^T F, eq. 28, with J square), we
remove one joint's mobility rather than leaving it uncontrolled. Concretely:
joint 4 (elbow) is welded at its "ready pose" home value by folding the
locked rotation into the static body quaternion of fr3_link4, then deleting
the <joint> element entirely. Every geometry/inertial/child element defined
*inside* fr3_link4 is expressed in fr3_link4's own local frame and is
therefore untouched -- only the fixed transform from fr3_link3 to fr3_link4
changes, by exactly the rotation the joint would have contributed at that
fixed angle. This is Bhat & Bernstein (2000)a kinematically exact reduction, not an approximation.

Run with: uv run python scripts/build_model.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SRC_XML = ROOT / "assets" / "franka_fr3" / "fr3.xml"
OUT_ROBOT_XML = ROOT / "assets" / "franka_fr3" / "fr3_osc.xml"
# Kept in the same directory as fr3_osc.xml (not assets/) so that the
# compiler's meshdir="assets" keeps resolving correctly: MuJoCo does not
# honor an <include>d file's own <compiler> element, so a separate
# include-based scene one directory up silently drops meshdir. Merging
# everything into one self-contained file in franka_fr3/ sidesteps that.
OUT_SCENE_XML = ROOT / "assets" / "franka_fr3" / "fr3_osc_scene.xml"

LOCKED_JOINT = "fr3_joint4"

# The vendored Menagerie "home" keyframe (0,0,0,-1.57,0,1.57,-0.785) has
# joint2 = 0, a genuine kinematic singularity (no shoulder lift keeps
# joint3's axis coincident with joint1's regardless of joint4; verified
# numerically -- rank 5 of 6, smallest singular value ~1e-16). Switching to
# the well-known Franka "ready" configuration
# [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4] fixes that, but with joint4 welded
# that pose sits exactly on a *second*, independent singularity at joint5=0
# (also verified numerically the same way -- rank 5, min singular value
# ~1e-17, and confirmed via a joint-by-joint sweep that joint5=0 alone
# reproduces it regardless of the other joints). We therefore perturb joint5
# away from 0; -0.5 rad gives a comfortable margin (min singular value of J
# ~0.012 there vs ~1e-17 at joint5=0) while keeping the pose close to the
# standard ready configuration.
LOCKED_VALUE = -3 * np.pi / 4
READY_QPOS_6 = [0.0, -np.pi / 4, 0.0, -0.5, np.pi / 2, np.pi / 4]  # joints 1,2,3,5,6,7


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


def fold_joint_into_static_quat(body: ET.Element, joint: ET.Element, angle: float) -> None:
    """Replace body['quat'] with static_quat (x) joint_rotation(angle), in place."""
    raw = body.get("quat", "1 0 0 0")
    q_static = np.array([float(v) for v in raw.split()])
    q_static = q_static / np.linalg.norm(q_static)
    r_static = Rotation.from_quat(quat_wxyz_to_xyzw(q_static))

    axis = np.array([float(v) for v in joint.get("axis", "0 0 1").split()])
    r_joint = Rotation.from_rotvec(axis * angle)

    r_total = r_static * r_joint
    q_total_xyzw = r_total.as_quat()
    q_total_wxyz = quat_xyzw_to_wxyz(q_total_xyzw)
    body.set("quat", " ".join(f"{v:.10g}" for v in q_total_wxyz))


def main() -> None:
    tree = ET.parse(SRC_XML)
    root = tree.getroot()

    # --- 1. Weld fr3_joint4: fold its home rotation into fr3_link4's static quat.
    link4 = root.find(".//body[@name='fr3_link4']")
    assert link4 is not None
    joint4 = link4.find(f"joint[@name='{LOCKED_JOINT}']")
    assert joint4 is not None
    fold_joint_into_static_quat(link4, joint4, LOCKED_VALUE)
    link4.remove(joint4)

    # --- 2. Replace position actuators with direct joint-torque motors for the
    #     6 remaining joints, using each joint's own actuatorfrcrange as the
    #     motor's forcerange (drop the actuator for the now-welded joint4).
    actuator_block = root.find("actuator")
    assert actuator_block is not None
    frcrange = {}
    for j in root.iter("joint"):
        name = j.get("name")
        if name:
            frcrange[name] = j.get("actuatorfrcrange", "-1000 1000")

    for act in list(actuator_block):
        actuator_block.remove(act)
    active_joints = [f"fr3_joint{i}" for i in (1, 2, 3, 5, 6, 7)]
    for jname in active_joints:
        ET.SubElement(
            actuator_block,
            "motor",
            {
                "name": jname,
                "joint": jname,
                "ctrllimited": "true",
                "ctrlrange": frcrange[jname],
                "gear": "1",
            },
        )

    # --- 3. Add a small wiping-tool tip at the wrist attachment site, plus a
    #     wrist force/torque sensor there (reads the interaction force
    #     transmitted through fr3_joint7 -- i.e. everything downstream,
    #     including contact at the tool tip -- exactly like a real wrist F/T
    #     sensor).
    link7 = root.find(".//body[@name='fr3_link7']")
    assert link7 is not None
    attach_site = link7.find("site[@name='attachment_site']")
    assert attach_site is not None
    ET.SubElement(
        link7,
        "geom",
        {
            "name": "tool_tip",
            "type": "sphere",
            "size": "0.012",
            "pos": attach_site.get("pos", "0 0 0.107"),
            "rgba": "0.9 0.2 0.2 1",
            "group": "2",
            "friction": "1.0 0.01 0.0001",
            "condim": "4",
        },
    )

    sensor_block = root.find("sensor")
    if sensor_block is None:
        sensor_block = ET.SubElement(root, "sensor")
    ET.SubElement(sensor_block, "force", {"name": "wrist_force", "site": "attachment_site"})
    ET.SubElement(sensor_block, "torque", {"name": "wrist_torque", "site": "attachment_site"})

    # --- 4. Replace the keyframe with the non-singular ready pose (see
    #     LOCKED_VALUE comment above), in joint order 1,2,3,5,6,7.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        key = keyframe.find("key")
        key.set("qpos", " ".join(f"{v:.6g}" for v in READY_QPOS_6))
        key.set("ctrl", " ".join(f"{v:.6g}" for v in READY_QPOS_6))

    tree.write(OUT_ROBOT_XML, encoding="UTF-8", xml_declaration=False)
    print(f"Wrote {OUT_ROBOT_XML}")

    # --- 5. Scene: robot + floor + a table for the surface-wiping demo,
    #     merged into the SAME tree (see OUT_SCENE_XML comment above for why).
    root.set("model", "fr3 osc scene")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.35 0.35 0.35", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "130", "elevation": "-25"})
    ET.SubElement(root, "statistic", {"center": "0.4 0 0.3", "extent": "1.0"})

    asset = root.find("asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0",
        "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
        "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane", "texuniform": "true",
        "texrepeat": "5 5", "reflectance": "0.2",
    })
    ET.SubElement(asset, "material", {"name": "table_mat", "rgba": "0.55 0.42 0.3 1"})

    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0.3 0 1.6", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})
    # wiping-demo table: top surface at z = 0.475 m. Positioned so the ready
    # pose's EE (x~0.307-0.38 depending on the shoulder angle used to reach
    # down to this height) sits well inside the table's near edge. The
    # height (rather than the more "natural" ~0.42 m table height) is chosen
    # so the hybrid-control demo's approach configuration (joint2 = -0.45,
    # see demo_hybrid.py) stays reasonably well conditioned: min singular
    # value of J ~0.0077 there, vs ~0.0055 at the joint2 needed to reach a
    # lower ~0.42 m table -- see scripts/build_model.py git history / README
    # for the conditioning sweep this was picked from.
    table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0.45 0 0.265"})
    ET.SubElement(table, "geom", {
        "name": "table_top", "type": "box", "size": "0.28 0.28 0.21", "material": "table_mat",
        "friction": "1.0 0.01 0.0001",
        # Compliant contact (soft mat, not bare rigid steel): the default
        # MuJoCo contact stiffness is impulsive enough that our controller
        # (P-only force feedback + damping, no impact-transition control --
        # Sec. VIII-adjacent eq. 48 is explicitly out of scope) cannot hold a
        # stable contact force against it. A softer solref/solimp is a
        # standard simulation-side technique for contact-rich force control
        # and does not change the control law itself.
        "solref": "0.05 1", "solimp": "0.9 0.95 0.01",
    })

    tree.write(OUT_SCENE_XML, encoding="UTF-8", xml_declaration=False)
    print(f"Wrote {OUT_SCENE_XML}")


if __name__ == "__main__":
    main()
