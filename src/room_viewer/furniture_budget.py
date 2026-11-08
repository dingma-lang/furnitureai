"""Private, layout-specific financial assumptions. Integer cents; no price/ownership inference."""

import copy
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from uuid import uuid4

from .inventory import identifier, text, url
from .photo_projects import ID, now

ACTIONS = ("none", "retain", "move", "sell", "replace", "purchase_additional", "dispose_donate")
STATUSES = ("unknown", "estimate", "quote", "user_entered", "confirmed_purchase", "recorded_sale")
EXPENSE_KINDS = (
    "moving_company",
    "truck_rental",
    "fuel",
    "packing",
    "disassembly",
    "reassembly",
    "transport_labor",
    "storage",
    "other_moving",
    "other_scenario",
    "disposal_donation",
)
MAX_CENTS = 10**10
MAX_TOTAL = 10**12
MONEY = re.compile(r"^(0|[1-9]\d{0,8})(\.\d{1,2})?$")
RATE = re.compile(r"^(0|[1-9]\d{0,2})(\.\d{1,6})?$")


def uid():
    return uuid4().hex


def cents(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not MONEY.fullmatch(value.strip()):
        raise ValueError("Enter USD as digits with at most two decimals; blank means unknown")
    result = int(Decimal(value.strip()) * 100)
    if result > MAX_CENTS:
        raise ValueError("Amount exceeds prototype limit")
    return result


def money(value, status, source, timestamp=None):
    amount = cents(value)
    if status not in STATUSES:
        raise ValueError("Invalid financial evidence status")
    source = text(source, 2000)
    if amount is not None and (not source or status == "unknown"):
        raise ValueError("Known amounts require a source and evidence classification")
    return {
        "cents": amount,
        "currency": "USD",
        "status": status if amount is not None else "unknown",
        "source": source,
        "recorded_at": timestamp or now(),
        "independently_verified": False,
    }


def rate(value, jurisdiction, source, timestamp=None):
    if value is None or value == "":
        percent = None
    else:
        if not isinstance(value, str) or not RATE.fullmatch(value.strip()) or Decimal(value) > 100:
            raise ValueError(
                "Tax percent must be plain decimal 0–100, at most six fractional digits"
            )
        percent = value.strip()
        if not text(jurisdiction, 300) or not text(source):
            raise ValueError("Describe tax jurisdiction and assumption source")
    return {
        "percent": percent,
        "jurisdiction": text(jurisdiction, 300),
        "source": text(source),
        "recorded_at": timestamp or now(),
        "status": "user_assumption",
        "independently_verified": False,
    }


def cents_valid(value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_CENTS
    ):
        raise ValueError("Amounts must be nonnegative bounded integer cents or null")


def validate_money(record):
    cents_valid(record["cents"])
    if (
        record.get("currency") != "USD"
        or record.get("status") not in STATUSES
        or record.get("independently_verified") is not False
    ):
        raise ValueError("Unsupported currency or financial verification claim")
    if not isinstance(record.get("recorded_at"), str) or not record["recorded_at"]:
        raise ValueError("Missing amount timestamp")
    if record["cents"] is not None and (
        not text(record.get("source", "")) or record["status"] == "unknown"
    ):
        raise ValueError("Missing financial provenance")


def boolean_unknown(value):
    if value is not None and not isinstance(value, bool):
        raise ValueError("Tax applicability must be true, false or unknown")
    return value


def empty_scenario(pid, lid):
    return {
        "id": uid(),
        "destination_project_id": pid,
        "layout_id": lid,
        "name": "Furniture budget",
        "currency": "USD",
        "created_at": now(),
        "modified_at": now(),
        "item_actions": {},
        "purchases": {},
        "sales": {},
        "expenses": {},
        "notes": "",
        "history": [],
        "redo": [],
        "ownership_changes_applied": False,
    }


def validate_budget(budget, pid):
    if (
        budget.get("schema_version") != 1
        or budget.get("currency") != "USD"
        or not isinstance(budget.get("scenarios"), dict)
        or len(budget["scenarios"]) > 100
    ):
        raise ValueError("Unsupported or oversized financial schema")
    scenario_ids = set()
    for lid, s in budget["scenarios"].items():
        if (
            not ID.fullmatch(lid)
            or s["layout_id"] != lid
            or s["destination_project_id"] != pid
            or not ID.fullmatch(s["id"])
            or s["id"] in scenario_ids
            or s.get("currency") != "USD"
            or s.get("ownership_changes_applied") is not False
        ):
            raise ValueError("Invalid scenario identity/currency/ownership claim")
        scenario_ids.add(s["id"])
        events = set()
        if sum(len(s[k]) for k in ("purchases", "sales", "expenses")) > 500:
            raise ValueError("Scenario has too many financial records")
        for iid, record in s["item_actions"].items():
            identifier(iid)
            if record["action"] not in ACTIONS or not text(record["source"]):
                raise ValueError("Invalid financial action")
        for collection in ("purchases", "sales", "expenses"):
            for key, r in s[collection].items():
                if not ID.fullmatch(r["id"]) or r["id"] in events:
                    raise ValueError("Duplicate financial event ID")
                events.add(r["id"])
                if not text(r.get("source", "")) or not isinstance(r.get("modified_at"), str):
                    raise ValueError("Describe financial record provenance")
                if collection == "purchases":
                    if (
                        key != r["id"]
                        or r.get("proposed_replacement_id") != r["id"]
                        or not text(r["description"], 500)
                    ):
                        raise ValueError("Invalid proposed purchase")
                    if r.get("replacement_of") is not None:
                        identifier(r["replacement_of"])
                    if (
                        isinstance(r["quantity"], bool)
                        or not isinstance(r["quantity"], int)
                        or not 1 <= r["quantity"] <= 1000
                    ):
                        raise ValueError("Quantity must be 1–1000")
                    if r.get("product_url"):
                        url(r["product_url"])
                    if r.get("catalog_candidate") is not None:
                        candidate = r["catalog_candidate"]
                        if (
                            not isinstance(candidate, dict)
                            or not text(candidate.get("product_id", ""), 300)
                            or candidate.get("exact_identity_verified") is not False
                        ):
                            raise ValueError(
                                "Catalog candidates must remain explicitly unverified references"
                            )
                    validate_money(r["unit_price"])
                    validate_money(r["discount"])
                    boolean_unknown(r["goods_taxable"])
                    tax = r["tax"]
                    rebuilt = rate(
                        tax["percent"], tax["jurisdiction"], tax["source"], tax["recorded_at"]
                    )
                    if (
                        tax.get("independently_verified") is not False
                        or rebuilt["percent"] != tax["percent"]
                    ):
                        raise ValueError("Invalid tax provenance")
                    for fee in r["fees"].values():
                        validate_money(fee["amount"])
                        boolean_unknown(fee["taxable"])
                    if set(r["fees"]) != {"delivery", "assembly", "other"}:
                        raise ValueError("Missing purchase fee categories")
                    if (
                        r["unit_price"]["cents"] is not None
                        and r["discount"]["cents"] is not None
                        and r["discount"]["cents"] > r["quantity"] * r["unit_price"]["cents"]
                    ):
                        raise ValueError("Discount exceeds purchase subtotal")
                    if (
                        r["unit_price"]["cents"] is not None
                        and r["quantity"] * r["unit_price"]["cents"] > MAX_CENTS
                    ):
                        raise ValueError("Purchase subtotal exceeds limit")
                elif collection == "sales":
                    identifier(key)
                    if key != r["inventory_item_id"] or r["state"] not in (
                        "proposed",
                        "completed_recorded",
                    ):
                        raise ValueError("Invalid sale reference/state")
                    boolean_unknown(r.get("seller_costs_prepaid"))
                    for m in r["prices"].values():
                        validate_money(m)
                    if set(r["prices"]) != {
                        "conservative",
                        "expected",
                        "optimistic",
                        "asking",
                        "actual",
                    }:
                        raise ValueError("Missing resale classifications")
                    for costs in (r["estimated_costs"], r["actual_costs"]):
                        if set(costs) != {"selling_fees", "pickup", "other"}:
                            raise ValueError("Missing seller expense categories")
                        for m in costs.values():
                            validate_money(m)
                    if r["state"] == "completed_recorded":
                        if r.get("actual_confirmed") is not True:
                            raise ValueError(
                                "Confirm a recorded completed sale; no transaction is processed"
                            )
                        date.fromisoformat(r["completed_date"])
                    low, expected, high = (
                        r["prices"][k]["cents"] for k in ("conservative", "expected", "optimistic")
                    )
                    if any(
                        a is not None and b is not None and a > b
                        for a, b in ((low, expected), (expected, high), (low, high))
                    ):
                        raise ValueError("Resale estimates must be ordered low ≤ expected ≤ high")
                else:
                    if (
                        key != r["id"]
                        or r["category"] not in EXPENSE_KINDS
                        or r["scope"] not in ("scenario", "item")
                    ):
                        raise ValueError("Invalid expense category/scope")
                    if r["scope"] == "item":
                        identifier(r["inventory_item_id"])
                    elif r.get("inventory_item_id") is not None:
                        raise ValueError("Scenario expense cannot also be an item charge")
                    for iid in r["covers_item_ids"]:
                        identifier(iid)
                    if len(set(r["covers_item_ids"])) != len(r["covers_item_ids"]):
                        raise ValueError("Duplicate covered item")
                    validate_money(r["amount"])
    return budget


def remap_budget(project):
    if project.get("furniture_budget"):
        for scenario in project["furniture_budget"]["scenarios"].values():
            scenario["destination_project_id"] = project["id"]
            scenario["history"] = []
            scenario["redo"] = []
        validate_budget(project["furniture_budget"], project["id"])


def duplicate_scenario(project, old_layout, new_layout):
    budget = project.get("furniture_budget")
    if budget and old_layout in budget["scenarios"]:
        old = budget["scenarios"][old_layout]
        new = copy.deepcopy(old)
        new.update(
            id=uid(),
            layout_id=new_layout,
            cloned_from_scenario_id=old["id"],
            created_at=now(),
            modified_at=now(),
            history=[],
            redo=[],
        )
        budget["scenarios"][new_layout] = new


def part(value=0, missing=None):
    return {
        "cents": value if not missing else None,
        "known_cents": value,
        "missing": sorted(set(missing or [])),
    }


def combine(values):
    value = sum(v["known_cents"] for v in values)
    missing = [m for v in values for m in v["missing"]]
    if abs(value) > MAX_TOTAL:
        raise ValueError("Financial scenario total exceeds prototype limit")
    return part(value, missing)


def field(record, path):
    return part(record["cents"] or 0, [path] if record["cents"] is None else [])


def minus(first, *rest):
    return combine([first, *[part(-v["known_cents"], v["missing"]) for v in rest]])


def tax_for(amount, applicable, tax, path):
    if amount["cents"] == 0 or applicable is False:
        return part()
    if amount["cents"] is None:
        return part(0, [path + ".tax_base"])
    if applicable is None:
        return part(0, [path + ".tax_applicability"])
    if tax["percent"] is None:
        return part(0, [path + ".tax_rate"])
    result = int(
        (Decimal(amount["cents"]) * Decimal(tax["percent"]) / 100).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    return part(result)


def calculate(scenario, layout, inventory):
    purchases = []
    taxes = []
    fees = []
    moving = []
    other = []
    expected = []
    optimistic = []
    conservative_values = []
    realized = []
    prepaid_seller_costs = []
    action_missing = []
    uncertainty = []
    relevant = (
        {
            p.get("inventory_item_id") or p.get("replaces_inventory_item_id")
            for p in layout.get("placements", [])
            if p.get("inventory_item_id") or p.get("replaces_inventory_item_id")
        }
        | set(layout.get("decisions", {}))
        | set(scenario["item_actions"])
        | {p["replacement_of"] for p in scenario["purchases"].values() if p.get("replacement_of")}
        | set(scenario["sales"])
        | {
            e["inventory_item_id"]
            for e in scenario["expenses"].values()
            if e.get("inventory_item_id")
        }
    )
    for p in layout.get("replacement_planning", {}).get("proposals", {}).values():
        if p["status"] == "applied" and not p.get("purchase_event_id"):
            action_missing.append("replacement_purchase_link." + p["id"])
    for iid in relevant:
        action = scenario["item_actions"].get(iid, {}).get("action", "none")
        if action == "none":
            action_missing.append("item_actions." + iid)
        if action == "retain" and (
            iid in scenario["sales"]
            or any(p.get("replacement_of") == iid for p in scenario["purchases"].values())
        ):
            action_missing.append("retain_conflicts_with_saved_purchase_or_sale." + iid)

        if iid not in inventory:
            action_missing.append("inventory_reference." + iid)
        if action in ("replace", "purchase_additional") and not any(
            p.get("replacement_of") == iid for p in scenario["purchases"].values()
        ):
            action_missing.append("purchase_required." + iid)
        if action == "sell" and iid not in scenario["sales"]:
            action_missing.append("sale_required." + iid)
        if action in ("move", "dispose_donate") and not any(
            (e["scope"] == "item" and e["inventory_item_id"] == iid)
            or (e["scope"] == "scenario" and iid in e["covers_item_ids"])
            for e in scenario["expenses"].values()
        ):
            action_missing.append("item_expense_required." + iid)
    for pid, p in scenario["purchases"].items():
        path = "purchases." + pid
        subtotal = minus(
            part(
                (p["unit_price"]["cents"] or 0) * p["quantity"],
                [path + ".unit_price"] if p["unit_price"]["cents"] is None else [],
            ),
            field(p["discount"], path + ".discount"),
        )
        purchases.append(subtotal)
        taxes.append(tax_for(subtotal, p["goods_taxable"], p["tax"], path + ".goods"))
        for name, fee in p["fees"].items():
            cost = field(fee["amount"], path + "." + name)
            fees.append(cost)
            taxes.append(tax_for(cost, fee["taxable"], p["tax"], path + "." + name))
        if p["unit_price"]["status"] != "confirmed_purchase":
            uncertainty.append(path + ".purchase_not_confirmed")
    for eid, e in scenario["expenses"].items():
        (other if e["category"] in ("other_scenario", "disposal_donation") else moving).append(
            field(e["amount"], "expenses." + eid)
        )
        if e["amount"]["status"] not in ("confirmed_purchase", "recorded_sale"):
            uncertainty.append("expenses." + eid + ".unconfirmed")
    for iid, sale in scenario["sales"].items():
        path = "sales." + iid
        if sale["state"] == "completed_recorded":
            realized.append(
                minus(
                    field(sale["prices"]["actual"], path + ".actual"),
                    combine(
                        [
                            field(m, path + ".actual_costs." + k)
                            for k, m in sale["actual_costs"].items()
                        ]
                    ),
                )
            )
        else:
            expenses = combine(
                [
                    field(m, path + ".estimated_costs." + k)
                    for k, m in sale["estimated_costs"].items()
                ]
            )
            if sale.get("seller_costs_prepaid") is True:
                prepaid_seller_costs.append(expenses)
            elif sale.get("seller_costs_prepaid") is None and expenses["cents"] != 0:
                prepaid_seller_costs.append(part(0, [path + ".seller_expense_timing"]))
            expected.append(minus(field(sale["prices"]["expected"], path + ".expected"), expenses))
            optimistic.append(
                minus(field(sale["prices"]["optimistic"], path + ".optimistic"), expenses)
            )
            conservative_values.append(
                minus(field(sale["prices"]["conservative"], path + ".conservative"), expenses)
            )
            uncertainty.append(path + ".future_sale_not_guaranteed")
    totals = {
        "purchase_subtotal": combine(purchases),
        "taxes": combine(taxes),
        "purchase_fees": combine(fees),
        "moving_costs": combine(moving),
        "other_expenses": combine(other),
        "future_expected_resale": combine(expected),
        "future_optimistic_resale": combine(optimistic),
        "future_conservative_resale_reference": combine(conservative_values),
        "realized_net_resale": combine(realized),
    }
    gross = combine(
        [
            totals[k]
            for k in (
                "purchase_subtotal",
                "taxes",
                "purchase_fees",
                "moving_costs",
                "other_expenses",
            )
        ]
        + [part(0, action_missing)]
    )
    totals["gross_upfront"] = gross
    totals["cash_before_future_sales"] = combine([gross, combine(prepaid_seller_costs)])
    totals["conservative_net"] = minus(
        totals["cash_before_future_sales"], totals["realized_net_resale"]
    )
    totals["expected_net"] = minus(
        gross, totals["realized_net_resale"], totals["future_expected_resale"]
    )
    totals["optimistic_net"] = minus(
        gross, totals["realized_net_resale"], totals["future_optimistic_resale"]
    )
    missing = sorted({m for p in totals.values() for m in p["missing"]})
    return {
        "currency": "USD",
        "totals": totals,
        "missing_assumptions": missing,
        "missing_count": len(missing),
        "uncertainty": "incomplete"
        if missing
        else "estimates_present"
        if uncertainty
        else "user_recorded_inputs",
        "uncertainty_reasons": uncertainty,
        "physical_placement_verified": False,
        "notice": "User-entered financial assumptions, not live quotes or official tax advice. Future resale is not guaranteed cash. Historical purchase prices are reference only.",
    }


class BudgetStore:
    def __init__(self, transfer):
        self.transfer = transfer
        self.projects = transfer.projects
        self.inventory = transfer.inventory
        self.lock = transfer.lock

    def read(self, pid):
        project = self.transfer.read(pid)
        if project.get("furniture_budget"):
            validate_budget(project["furniture_budget"], pid)
        return project

    def public(self, project):
        home = project["home_transfer"]
        budget = project.get("furniture_budget") or {
            "schema_version": 1,
            "currency": "USD",
            "scenarios": {},
        }
        library = self.inventory.listing()["items"]
        comparison = {
            lid: {
                "layout_name": layout["name"],
                "scenario_id": budget["scenarios"].get(lid, {}).get("id"),
                "calculation": calculate(budget["scenarios"][lid], layout, library)
                if lid in budget["scenarios"]
                else None,
            }
            for lid, layout in home["layouts"].items()
        }
        inventory = {
            iid: {
                k: item.get(k)
                for k in (
                    "inventory_item_id",
                    "display_name",
                    "category",
                    "condition",
                    "identification",
                    "specifications",
                )
            }
            for iid, item in library.items()
        }
        return {
            "project_id": project["id"],
            "project_revision": project["revision"],
            "budget": copy.deepcopy(budget),
            "layouts": {
                lid: {"name": v["name"], "decisions": v["decisions"], "placements": v["placements"]}
                for lid, v in home["layouts"].items()
            },
            "active_layout": home["active_layout"],
            "historical_layout_scenarios": sorted(set(budget["scenarios"]) - set(home["layouts"])),
            "comparison": comparison,
            "inventory": inventory,
            "physical_placement_verified": False,
        }

    def get(self, pid):
        with self.lock:
            return self.public(self.read(pid))

    def action(self, pid, body):
        with self.lock:
            project = self.read(pid)
            lid = body["layout_id"]
            if lid not in project["home_transfer"]["layouts"]:
                raise ValueError("Select an existing layout")
            budget = project.setdefault(
                "furniture_budget", {"schema_version": 1, "currency": "USD", "scenarios": {}}
            )
            op = body["operation"]
            if op == "ensure":
                if lid not in budget["scenarios"]:
                    budget["scenarios"][lid] = empty_scenario(pid, lid)
                else:
                    return self.public(project)
            else:
                if lid not in budget["scenarios"]:
                    raise ValueError("Open a scenario first")
                s = budget["scenarios"][lid]
                if op in ("undo", "redo"):
                    source = s["history"] if op == "undo" else s["redo"]
                    target = s["redo"] if op == "undo" else s["history"]
                    if not source:
                        raise ValueError("No financial operation to " + op)
                    snapshot = {
                        k: copy.deepcopy(v) for k, v in s.items() if k not in ("history", "redo")
                    }
                    old = source.pop()
                    target.append(snapshot)
                    history, redo = s["history"], s["redo"]
                    s = old
                    s.update(history=history, redo=redo)
                    budget["scenarios"][lid] = s
                else:
                    before = {
                        k: copy.deepcopy(v) for k, v in s.items() if k not in ("history", "redo")
                    }
                    timestamp = now()
                    record = body.get("record", {})
                    source = text(record.get("source", body.get("source", "")), 2000)
                    if op == "item_action":
                        iid = body["inventory_item_id"]
                        self.inventory.get(self.inventory.read(), iid)
                        s["item_actions"][iid] = {
                            "action": body["action"],
                            "source": text(body.get("source", "Explicit user planning decision")),
                            "recorded_at": timestamp,
                        }
                    elif op == "purchase":
                        event_id = record.get("id")
                        if not isinstance(event_id, str) or not ID.fullmatch(event_id):
                            raise ValueError(
                                "A stable purchase event ID is required for idempotent save"
                            )
                        status = record.get("price_status", "estimate")
                        replacement = record.get("replacement_of") or None
                        if replacement:
                            self.inventory.get(self.inventory.read(), replacement)

                        def m(value):
                            return money(value, status, source, timestamp)

                        s["purchases"][event_id] = {
                            "id": event_id,
                            "description": text(record.get("description", ""), 500),
                            "replacement_of": replacement,
                            "retailer": text(record.get("retailer", ""), 300),
                            "product_url": url(record["product_url"])
                            if record.get("product_url")
                            else None,
                            "catalog_candidate": copy.deepcopy(record.get("catalog_candidate")),
                            "quantity": record.get("quantity", 1),
                            "unit_price": m(record.get("unit_price")),
                            "discount": m(record.get("discount")),
                            "goods_taxable": boolean_unknown(record.get("goods_taxable")),
                            "tax": rate(
                                record.get("tax_percent"),
                                record.get("jurisdiction", ""),
                                source,
                                timestamp,
                            ),
                            "fees": {
                                key: {
                                    "amount": m(record.get(key)),
                                    "taxable": boolean_unknown(record.get(key + "_taxable")),
                                }
                                for key in ("delivery", "assembly", "other")
                            },
                            "notes": text(record.get("notes", ""), 10000),
                            "source": source,
                            "modified_at": timestamp,
                            "proposed_replacement_id": event_id,
                            "future": {"manufacturer_price_reference": None},
                        }
                    elif op == "sale":
                        iid = record["inventory_item_id"]
                        self.inventory.get(self.inventory.read(), iid)
                        prior = s["sales"].get(iid, {})
                        prices = {
                            key: money(
                                record.get(key),
                                "recorded_sale" if key == "actual" else "estimate",
                                source,
                                timestamp,
                            )
                            for key in (
                                "conservative",
                                "expected",
                                "optimistic",
                                "asking",
                                "actual",
                            )
                        }
                        s["sales"][iid] = {
                            "id": prior.get("id", record.get("id") or uid()),
                            "inventory_item_id": iid,
                            "state": record.get("state", "proposed"),
                            "seller_costs_prepaid": boolean_unknown(
                                record.get("seller_costs_prepaid")
                            ),
                            "listing_status": text(record.get("listing_status", "not_listed"), 120),
                            "prices": prices,
                            "estimated_costs": {
                                k: money(record.get(k), "estimate", source, timestamp)
                                for k in ("selling_fees", "pickup", "other")
                            },
                            "actual_costs": {
                                k: money(
                                    record.get("actual_" + k), "recorded_sale", source, timestamp
                                )
                                for k in ("selling_fees", "pickup", "other")
                            },
                            "actual_confirmed": record.get("actual_confirmed") is True,
                            "completed_date": record.get("completed_date") or None,
                            "source": source,
                            "modified_at": timestamp,
                            "condition_notes": text(record.get("condition_notes", ""), 3000),
                            "notes": text(record.get("notes", ""), 10000),
                            "future": {
                                "marketplace_listing_id": None,
                                "negotiated_price": None,
                                "pickup_terms": text(record.get("pickup_terms", ""), 2000),
                                "transaction_processed": False,
                            },
                        }
                    elif op == "expense":
                        eid = record.get("id")
                        if not isinstance(eid, str) or not ID.fullmatch(eid):
                            raise ValueError(
                                "A stable expense event ID is required for idempotent save"
                            )
                        iid = record.get("inventory_item_id") or None
                        if iid:
                            self.inventory.get(self.inventory.read(), iid)
                        for covered in record.get("covers_item_ids", []):
                            self.inventory.get(self.inventory.read(), covered)
                        s["expenses"][eid] = {
                            "id": eid,
                            "description": text(record.get("description", ""), 500),
                            "category": record.get("category", "other_moving"),
                            "scope": record.get("scope", "scenario"),
                            "inventory_item_id": iid,
                            "covers_item_ids": record.get("covers_item_ids", []),
                            "amount": money(
                                record.get("amount"),
                                record.get("status", "estimate"),
                                source,
                                timestamp,
                            ),
                            "source": source,
                            "notes": text(record.get("notes", ""), 10000),
                            "modified_at": timestamp,
                            "future": {"vehicle_calculation": None},
                        }
                    elif op == "remove":
                        if body["collection"] not in ("purchases", "sales", "expenses"):
                            raise ValueError("Unknown financial collection")
                        if body["record_id"] not in s[body["collection"]]:
                            raise ValueError("Financial record not found")
                        del s[body["collection"]][body["record_id"]]
                    else:
                        raise ValueError("Unknown financial action")
                    s["history"] = (s["history"] + [before])[-40:]
                    s["redo"] = []
                s["modified_at"] = now()
            validate_budget(budget, pid)
            from .replacements import validate_project

            validate_project(project)
            # Evaluate before committing to catch totals overflow and malformed arithmetic.
            result = self.public(project)
            self.projects.write(project)
            result["project_revision"] = project["revision"]
            return result
