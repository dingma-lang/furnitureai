"""Prepare only a previously approved local ARKitScenes scene; no downloads."""

import argparse

from room_multiview.arkitscenes import prepare_scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "output", "scene-id", "approval-record"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--sky-direction", choices=["Up", "Left", "Down", "Right"], required=True)
    p.add_argument("--tracking-step", type=int, default=6)
    p.add_argument("--primary-step", type=int, default=180)
    p.add_argument("--start-index", type=int, default=60)
    p.add_argument("--stop-index", type=int, default=5580)
    a = p.parse_args()
    print(
        prepare_scene(
            a.assets,
            a.output,
            a.scene_id,
            a.sky_direction,
            a.approval_record,
            a.start_index,
            a.stop_index,
            a.tracking_step,
            a.primary_step,
        )
    )


if __name__ == "__main__":
    main()
