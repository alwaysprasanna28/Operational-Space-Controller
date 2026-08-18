"""Thin MuJoCo model/data loading helpers for the 6-DOF FR3 used throughout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
SCENE_XML = ASSETS_DIR / "franka_fr3" / "fr3_osc_scene.xml"
ROBOT_ONLY_XML = ASSETS_DIR / "franka_fr3" / "fr3_osc.xml"

EE_SITE = "attachment_site"


@dataclass
class Robot:
    """A MuJoCo model/data pair plus the handles OSC code needs repeatedly.

    ``scratch`` is a second, independent MjData used only to evaluate g(q)
    at zero velocity (see dynamics.gravity) without disturbing the live
    simulation state in ``data``.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    scratch: mujoco.MjData
    ee_site_id: int

    @property
    def nv(self) -> int:
        return self.model.nv

    def home_qpos(self) -> np.ndarray:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        return self.model.key_qpos[key_id].copy()

    def reset_home(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def ee_rot(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()


def load_robot(xml_path: Path | str = SCENE_XML) -> Robot:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if ee_site_id < 0:
        raise ValueError(f"site '{EE_SITE}' not found in {xml_path}")
    robot = Robot(model=model, data=data, scratch=scratch, ee_site_id=ee_site_id)
    robot.reset_home()
    return robot
