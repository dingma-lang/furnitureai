"""Read-only, model-free project discovery and explicit review labels."""

from furniture_bench.data import read_json

from .photo_projects import now


def project_progress(project, library):
    photos = [p for p in project.get("photos", []) if not p.get("deleted")]
    home = project.get("home_transfer")
    steps = []
    warnings = []
    if home:
        if home.get("floorplan"):
            steps.append("Floor plan added")
        if home.get("rooms"):
            steps.append("Room layout started")
        if any(layout.get("placements") for layout in home["layouts"].values()):
            steps.append("Furniture placed in a design")
        if any(
            layout.get("replacement_planning", {}).get("proposals")
            for layout in home["layouts"].values()
        ):
            steps.append("Replacement proposals available")
        if project.get("furniture_budget", {}).get("scenarios"):
            steps.append("Budget started")
        kind = "moving_plan"
    else:
        if photos:
            steps.append("Photo uploaded")
        if any(
            d.get("review_status") != "rejected" and not d.get("confirmed")
            for p in photos
            for d in p.get("reviewed", [])
        ):
            steps.append("Furniture awaiting review")
        if any(
            d.get("confirmed") and d.get("review_status") == "confirmed"
            for p in photos
            for d in p.get("reviewed", [])
        ):
            steps.append("Furniture confirmed by user")
        if any(
            any(link.get("project_id") == project["id"] for link in item.get("links", []))
            for item in library.get("items", {}).values()
        ):
            steps.append("Furniture saved")
        kind = "photo_project"
    return {
        "id": project["id"],
        "name": project["name"],
        "type": kind,
        "state": steps[-1] if steps else "Project created",
        "steps": steps,
        "updated_at": project.get("updated_at"),
        "revision": project.get("revision", 0),
        "physical_placement_verified": False,
        "warnings": warnings,
    }


def dashboard(projects, inventory):
    """No model import/load, geometry mutation or historical project migration."""
    with projects.lock:
        library = inventory.listing()
        items = []
        unavailable = 0
        for path in sorted(
            projects.root.glob("*/project.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                project = read_json(path)
                item = project_progress(project, library)
                for photo in project.get("photos", []):
                    if (
                        not photo.get("deleted")
                        and not (
                            projects.folder(project["id"])
                            / "photos"
                            / photo["id"]
                            / "processing.png"
                        ).is_file()
                    ):
                        item["warnings"].append(
                            "A source photograph is unavailable; saved reviews remain preserved"
                        )
                plan = (project.get("home_transfer") or {}).get("floorplan")
                if (
                    plan
                    and not (
                        projects.folder(project["id"]) / "floorplans" / plan["preview"]
                    ).is_file()
                ):
                    item["warnings"].append(
                        "Floor-plan preview unavailable; saved room geometry remains preserved"
                    )
                items.append(item)
            except (OSError, ValueError, KeyError, TypeError):
                unavailable += 1
        return {
            "schema_version": 1,
            "projects": items,
            "owned_count": sum(not i.get("archived") for i in library["items"].values()),
            "unavailable_projects": unavailable,
            "generated_at": now(),
            "model_initialization_required": False,
        }
