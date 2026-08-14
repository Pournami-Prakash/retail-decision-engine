import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from retail_decision_engine.service import DecisionHandler


class ServiceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DecisionHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health_separates_service_health_from_policy_clearance(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/health") as response:
            payload = json.load(response)
        self.assertEqual(payload["service"], "healthy")
        self.assertEqual(payload["decision_policy"], "blocked_until_release_gates_pass")

    def test_decision_endpoint_fails_closed(self) -> None:
        body = json.dumps(
            {
                "category": "cereal",
                "store": 8,
                "upc": 1600066590,
                "discount": 0.05,
                "replacement_unit_cost": 2.5,
                "supplier_funding_per_unit": 0.1,
                "inventory_available": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/decisions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 422)
        payload = json.load(caught.exception)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn(
            payload["reasons"][0],
            {"causal_gate_artifact_missing", "causal_estimate_not_cleared"},
        )


if __name__ == "__main__":
    unittest.main()
