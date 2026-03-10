import unittest
from onyx.streaming import OnyxStreamProducer

class TestOnyxStreaming(unittest.TestCase):
    def test_producer_initialization(self):
        producer = OnyxStreamProducer(host="127.0.0.1", port=9091)
        self.assertEqual(producer.host, "127.0.0.1")
        self.assertEqual(producer.port, 9091)
        self.assertIsNone(producer.sock)

if __name__ == "__main__":
    unittest.main()
