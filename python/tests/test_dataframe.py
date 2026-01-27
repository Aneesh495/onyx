import unittest
import onyx

class TestOnyxDataFrame(unittest.TestCase):
    def test_basic_sql_generation(self):
        df = onyx.DataFrame("telemetry")
        q = df.select("device_id", "latency_us").filter("temperature > 50.0").limit(5)
        self.assertEqual(
            q.to_sql(),
            "SELECT device_id, latency_us FROM telemetry WHERE temperature > 50.0 LIMIT 5;"
        )

    def test_default_select_all(self):
        df = onyx.DataFrame("events")
        self.assertEqual(df.to_sql(), "SELECT * FROM events;")

    def test_immutability_on_clone(self):
        base = onyx.DataFrame("sensors")
        filtered = base.filter("val > 10")
        self.assertNotEqual(base.to_sql(), filtered.to_sql())
        self.assertEqual(base.to_sql(), "SELECT * FROM sensors;")
        self.assertEqual(filtered.to_sql(), "SELECT * FROM sensors WHERE val > 10;")

if __name__ == "__main__":
    unittest.main()
