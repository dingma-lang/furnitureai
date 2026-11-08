"""Persist reviewed detection associations on existing editable furniture entities."""

from __future__ import annotations

import copy

from .model import validate_scene


def associate_observation(
    scene, observation, entity_id, reviewer, evidence, demo_only=False, contact_estimate=None
):
    from furniture_bench.room_links import link_observation

    validate_scene(scene)
    # Existing link validator checks category, grouping review, and explicit demo status.
    link_observation(observation, scene, entity_id, reviewer, evidence, demo_only=demo_only)
    if contact_estimate is not None:
        if (
            contact_estimate.get("status") != "estimated"
            or contact_estimate.get("units") != "m"
            or contact_estimate.get("source_image_sha256") != observation["image_sha256"]
            or contact_estimate.get("room_id") != scene["room_id"]
            or contact_estimate.get("entity_centre_pose") is not None
        ):
            raise ValueError(
                "contact estimate must match source image/room and must not claim an entity-centre pose"
            )
    updated = copy.deepcopy(scene)
    entity = next(item for item in updated["furniture"] if item["id"] == entity_id)
    references = entity.setdefault("photo_observations", [])
    if any(ref["observation_id"] == observation["observation_id"] for ref in references):
        raise ValueError("observation is already associated with this furniture entity")
    references.append(
        {
            "observation_id": observation["observation_id"],
            "image_id": observation["image_id"],
            "image_sha256": observation["image_sha256"],
            "image_frame": copy.deepcopy(observation["image_frame"]),
            "image_regions": copy.deepcopy(observation["image_regions"]),
            "association_status": "demo_only" if demo_only else "reviewed",
            "reviewer": reviewer,
            "evidence": evidence,
            "metric_geometry_applied": False,
            "ground_contact_estimate": copy.deepcopy(contact_estimate),
        }
    )
    link = link_observation(
        observation, updated, entity_id, reviewer, evidence, demo_only=demo_only
    )
    return updated, link
