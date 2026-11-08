import copy
import tempfile
import unittest
from uuid import uuid4

from room_viewer.furniture_budget import BudgetStore, cents, validate_budget
from room_viewer.home_transfer import TransferStore
from room_viewer.inventory import InventoryStore
from room_viewer.photo_projects import ProjectStore


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.projects = ProjectStore(self.tmp.name)
        self.inventory = InventoryStore(self.projects)
        self.transfer = TransferStore(self.projects, self.inventory)
        self.store = BudgetStore(self.transfer)
        self.project = self.transfer.create({"name": "Budget test destination"})
        self.pid = self.project["id"]
        self.lid = self.project["home_transfer"]["active_layout"]
        self.item = self.inventory.action(
            {"operation": "create", "display_name": "Unknown test sofa", "category": "sofa"}
        )["item"]
        self.iid = self.item["inventory_item_id"]
        self.act("ensure")

    def act(self, op, **kw):
        self.result = self.store.action(self.pid, {"operation": op, "layout_id": self.lid, **kw})
        return self.result

    def purchase(self, **extra):
        record = {
            "id": uuid4().hex,
            "description": "Illustrative sofa proposal",
            "replacement_of": self.iid,
            "quantity": 2,
            "unit_price": "100.10",
            "discount": "10.00",
            "goods_taxable": True,
            "tax_percent": "8.25",
            "jurisdiction": "Test-only assumed jurisdiction",
            "delivery": "20",
            "delivery_taxable": True,
            "assembly": "0",
            "assembly_taxable": False,
            "other": "0",
            "other_taxable": False,
            "price_status": "estimate",
            "source": "Software fixture, not actual market quote",
            **extra,
        }
        self.act("purchase", record=record)
        return record

    def sale(self, **extra):
        record = {
            "inventory_item_id": self.iid,
            "conservative": "50",
            "expected": "100",
            "optimistic": "150",
            "asking": "200",
            "selling_fees": "10",
            "pickup": "5",
            "other": "0",
            "state": "proposed",
            "seller_costs_prepaid": False,
            "source": "Software fixture, not resale prediction",
            **extra,
        }
        self.act("sale", record=record)
        return record

    def expense(self, **extra):
        record = {
            "id": uuid4().hex,
            "description": "Test moving cost",
            "category": "moving_company",
            "scope": "scenario",
            "covers_item_ids": [self.iid],
            "amount": "40",
            "status": "estimate",
            "source": "Software fixture",
            **extra,
        }
        self.act("expense", record=record)
        return record

    def action(self, value):
        self.act("item_action", inventory_item_id=self.iid, action=value)

    def calc(self):
        return self.result["comparison"][self.lid]["calculation"]

    def total(self, key):
        return self.calc()["totals"][key]["cents"]

    def test_money_decimal_precision_and_zero(self):
        self.assertEqual(10, cents("0.10"))
        self.assertEqual(0, cents("0"))
        self.assertIsNone(cents(""))
        self.assertIsNone(cents(None))
        for value in (
            "-1",
            "1e3",
            "1,000",
            "0.001",
            "NaN",
            "Infinity",
            "01",
            True,
            0,
            1.2,
            "999999999.99",
        ):
            with self.assertRaises(ValueError):
                cents(value)

    def test_quantity_discount_tax_and_fees(self):
        self.action("replace")
        self.purchase()
        self.assertEqual(19020, self.total("purchase_subtotal"))
        self.assertEqual(1734, self.total("taxes"))
        self.assertEqual(2000, self.total("purchase_fees"))
        self.assertEqual(22754, self.total("gross_upfront"))

    def test_exact_small_amount_multiplication(self):
        self.action("replace")
        self.purchase(
            quantity=3, unit_price="0.10", discount="0", goods_taxable=False, delivery="0"
        )
        self.assertEqual(30, self.total("purchase_subtotal"))
        self.assertEqual(30, self.total("expected_net"))

    def test_taxable_vs_nontaxable_shipping(self):
        self.action("replace")
        p = self.purchase(delivery_taxable=False)
        self.assertEqual(1569, self.total("taxes"))
        p["delivery_taxable"] = True
        self.act("purchase", record=p)
        self.assertEqual(1734, self.total("taxes"))

    def test_tax_rounds_half_up_per_line(self):
        self.action("replace")
        self.purchase(quantity=1, unit_price="0.05", discount="0", tax_percent="10", delivery="0")
        self.assertEqual(1, self.total("taxes"))

    def test_unknown_tax_and_fee_keep_totals_incomplete(self):
        self.action("replace")
        self.purchase(tax_percent=None)
        self.assertIsNone(self.total("taxes"))
        self.assertIsNone(self.total("gross_upfront"))
        self.assertGreater(self.calc()["missing_count"], 0)
        self.purchase(unit_price="", delivery="", goods_taxable=None)
        self.assertIsNone(self.total("purchase_subtotal"))
        self.assertIsNone(self.total("purchase_fees"))

    def test_zero_tax_free_goods_needs_no_rate(self):
        self.action("replace")
        self.purchase(
            quantity=1,
            unit_price="0",
            discount="0",
            delivery="0",
            tax_percent=None,
            goods_taxable=None,
        )
        self.assertEqual(0, self.total("taxes"))
        self.assertEqual(0, self.total("expected_net"))

    def test_resale_proceeds_and_upfront_are_separate(self):
        self.action("replace")
        self.purchase()
        self.sale()
        self.expense()
        self.assertEqual(26754, self.total("gross_upfront"))
        self.assertEqual(8500, self.total("future_expected_resale"))
        self.assertEqual(26754, self.total("conservative_net"))
        self.assertEqual(18254, self.total("expected_net"))
        self.assertEqual(13254, self.total("optimistic_net"))
        self.assertEqual(3500, self.total("future_conservative_resale_reference"))

    def test_asking_price_is_never_expected_proceeds(self):
        self.action("sell")
        self.sale(expected="", asking="99999")
        self.assertIsNone(self.total("future_expected_resale"))
        self.assertEqual(0, self.total("conservative_net"))

    def test_completed_sale_not_subtracted_twice(self):
        self.action("sell")
        self.sale(
            state="completed_recorded",
            actual_confirmed=True,
            completed_date="2026-10-01",
            actual="90",
            actual_selling_fees="5",
            actual_pickup="0",
            actual_other="0",
        )
        self.assertEqual(8500, self.total("realized_net_resale"))
        self.assertEqual(0, self.total("future_expected_resale"))
        self.assertEqual(-8500, self.total("expected_net"))
        self.assertEqual(-8500, self.total("conservative_net"))

    def test_completed_sale_requires_attestation_and_date(self):
        for extra in (
            {"actual_confirmed": False, "completed_date": "2026-10-01"},
            {"actual_confirmed": True, "completed_date": "bad"},
        ):
            with self.assertRaises(ValueError):
                self.sale(state="completed_recorded", **extra)

    def test_actual_seller_costs_can_exceed_revenue(self):
        self.action("sell")
        self.sale(
            state="completed_recorded",
            actual_confirmed=True,
            completed_date="2026-10-01",
            actual="0",
            actual_selling_fees="5",
            actual_pickup="0",
            actual_other="0",
        )
        self.assertEqual(-500, self.total("realized_net_resale"))
        self.assertEqual(500, self.total("expected_net"))

    def test_moving_shared_charge_counted_once(self):
        self.action("move")
        self.assertIsNone(self.total("gross_upfront"))
        e = self.expense()
        self.assertEqual(4000, self.total("moving_costs"))
        self.assertEqual(4000, self.total("gross_upfront"))
        self.act("expense", record=e)
        self.assertEqual(1, len(self.result["budget"]["scenarios"][self.lid]["expenses"]))
        self.assertEqual(4000, self.total("moving_costs"))

    def test_specific_item_expense_and_other_costs(self):
        self.action("move")
        self.expense(scope="item", inventory_item_id=self.iid, covers_item_ids=[])
        self.expense(category="other_scenario", amount="10", covers_item_ids=[])
        self.assertEqual(4000, self.total("moving_costs"))
        self.assertEqual(1000, self.total("other_expenses"))
        self.assertEqual(5000, self.total("gross_upfront"))

    def test_no_implicit_purchase_from_tentative_layout_decision(self):
        self.transfer.action(
            self.pid,
            {
                "operation": "decision",
                "inventory_item_id": self.iid,
                "status": "consider_replacing",
            },
        )
        r = self.store.get(self.pid)
        self.assertEqual({}, r["budget"]["scenarios"][self.lid]["purchases"])
        self.assertIsNone(
            r["comparison"][self.lid]["calculation"]["totals"]["gross_upfront"]["cents"]
        )
        self.action("retain")
        self.assertEqual(0, self.total("gross_upfront"))

    def test_missing_required_action_records(self):
        self.action("replace")
        self.assertIsNone(self.total("gross_upfront"))
        self.action("sell")
        self.assertIsNone(self.total("gross_upfront"))
        self.action("dispose_donate")
        self.assertIsNone(self.total("gross_upfront"))
        self.expense(category="disposal_donation", amount="0")
        self.assertEqual(0, self.total("gross_upfront"))

    def test_stable_purchase_id_prevents_repeat_save_duplicates(self):
        self.action("replace")
        p = self.purchase()
        self.act("purchase", record=p)
        self.assertEqual(1, len(self.result["budget"]["scenarios"][self.lid]["purchases"]))
        self.assertEqual(19020, self.total("purchase_subtotal"))
        del p["id"]
        with self.assertRaises(ValueError):
            self.act("purchase", record=p)

    def test_duplicate_event_across_collections_rejected(self):
        p = self.purchase()
        with self.assertRaises(ValueError):
            self.expense(id=p["id"])

    def test_discount_quantity_rate_validation_is_atomic(self):
        before = self.projects.read(self.pid)
        for extra in (
            {"quantity": -1},
            {"quantity": 0},
            {"quantity": True},
            {"quantity": 1.5},
            {"discount": "9999"},
            {"tax_percent": "101"},
            {"tax_percent": "1e2"},
            {"quantity": 1000, "unit_price": "99999999"},
        ):
            with self.assertRaises(ValueError):
                self.purchase(**extra)
        self.assertEqual(before, self.projects.read(self.pid))

    def test_resale_range_order(self):
        with self.assertRaises(ValueError):
            self.sale(conservative="200", expected="100")

    def test_layouts_are_independent_and_copy_is_explicit(self):
        self.action("replace")
        p = self.purchase()
        old = copy.deepcopy(self.result["budget"]["scenarios"][self.lid])
        old_lid = self.lid
        project = self.transfer.action(
            self.pid, {"operation": "alternative", "duplicate": True, "name": "Budget B"}
        )
        self.lid = project["home_transfer"]["active_layout"]
        self.act("ensure")
        new = self.result["budget"]["scenarios"][self.lid]
        self.assertNotEqual(new["id"], old["id"])
        p["unit_price"] = "150"
        self.act("purchase", record=p)
        self.assertEqual(old, self.result["budget"]["scenarios"][old_lid])
        self.assertEqual(29000, self.total("purchase_subtotal"))

    def test_financial_edits_preserve_inventory_and_geometry(self):
        before = self.inventory.path.read_bytes()
        home = copy.deepcopy(self.projects.read(self.pid)["home_transfer"])
        self.action("replace")
        self.purchase()
        self.sale()
        self.assertEqual(before, self.inventory.path.read_bytes())
        self.assertEqual(home, self.projects.read(self.pid)["home_transfer"])
        self.assertFalse(self.calc()["physical_placement_verified"])

    def test_historical_original_price_is_reference_only(self):
        self.inventory.action(
            {
                "operation": "edit",
                "inventory_item_id": self.iid,
                "condition": {"original_price": {"amount": 10000, "currency": "USD"}},
            }
        )
        self.action("retain")
        self.assertEqual(0, self.total("expected_net"))

    def test_undo_redo_reopen(self):
        self.action("replace")
        self.purchase()
        self.act("undo")
        self.assertEqual({}, self.result["budget"]["scenarios"][self.lid]["purchases"])
        self.act("redo")
        r = self.store.get(self.pid)
        self.assertEqual(
            19020, r["comparison"][self.lid]["calculation"]["totals"]["purchase_subtotal"]["cents"]
        )

    def test_geometry_undo_does_not_discard_budget(self):
        self.transfer.action(self.pid, {"operation": "clearance", "value": 0.8})
        self.action("replace")
        self.purchase()
        before = self.projects.read(self.pid)["furniture_budget"]
        self.transfer.action(self.pid, {"operation": "undo"})
        self.assertEqual(before, self.projects.read(self.pid)["furniture_budget"])

    def test_export_import_retains_ids_without_duplicate_events(self):
        self.action("replace")
        self.purchase()
        self.sale()
        raw = self.transfer.export(self.pid)
        new = self.transfer.import_bundle(raw)
        self.assertNotEqual(self.pid, new["id"])
        r = self.store.get(new["id"])
        s = r["budget"]["scenarios"][self.lid]
        self.assertEqual(new["id"], s["destination_project_id"])
        self.assertEqual(self.iid, list(s["purchases"].values())[0]["replacement_of"])
        self.assertEqual(1, len(s["purchases"]))
        self.assertEqual(1, len(self.inventory.read()["items"]))

    def test_schema_and_cents_tampering_rejected(self):
        b = copy.deepcopy(self.result["budget"])
        b["schema_version"] = 99
        with self.assertRaises(ValueError):
            validate_budget(b, self.pid)
        self.purchase()
        b = copy.deepcopy(self.result["budget"])
        r = next(iter(b["scenarios"][self.lid]["purchases"].values()))
        r["unit_price"]["cents"] = 1.2
        with self.assertRaises(ValueError):
            validate_budget(b, self.pid)

    def test_zero_vs_unknown_resale(self):
        self.action("sell")
        self.sale(
            conservative="0", expected="0", optimistic="0", selling_fees="0", pickup="0", other="0"
        )
        self.assertEqual(0, self.total("expected_net"))
        self.sale(conservative="", expected="", optimistic="")
        self.assertIsNone(self.total("expected_net"))
        self.assertEqual(0, self.total("conservative_net"))

    def test_multiple_items_shared_expense(self):
        other = self.inventory.action(
            {"operation": "create", "display_name": "Same-name test sofa", "category": "sofa"}
        )["item"]["inventory_item_id"]
        self.action("move")
        self.act("item_action", inventory_item_id=other, action="move")
        self.expense(covers_item_ids=[self.iid, other])
        self.assertEqual(4000, self.total("gross_upfront"))

    def test_standalone_manual_purchase_works_without_catalog_identity(self):
        self.purchase(replacement_of=None, description="Additional cabinet, unknown model")
        self.assertEqual(19020, self.total("purchase_subtotal"))
        self.assertIsNone(
            next(iter(self.result["budget"]["scenarios"][self.lid]["purchases"].values()))[
                "catalog_candidate"
            ]
        )

    def test_prepaid_seller_costs_affect_cash_but_are_not_double_counted(self):
        self.action("sell")
        self.sale(seller_costs_prepaid=True)
        self.assertEqual(1500, self.total("cash_before_future_sales"))
        self.assertEqual(1500, self.total("conservative_net"))
        self.assertEqual(-8500, self.total("expected_net"))

    def test_unknown_seller_expense_timing_leaves_cash_incomplete(self):
        self.action("sell")
        self.sale(seller_costs_prepaid=None)
        self.assertIsNone(self.total("cash_before_future_sales"))
        self.assertEqual(-8500, self.total("expected_net"))

    def test_retain_does_not_silently_resolve_conflicting_explicit_events(self):
        self.action("replace")
        self.purchase()
        self.action("retain")
        self.assertIsNone(self.total("expected_net"))
        self.assertTrue(any("retain_conflicts" in p for p in self.calc()["missing_assumptions"]))
        self.assertEqual(1, len(self.result["budget"]["scenarios"][self.lid]["purchases"]))
