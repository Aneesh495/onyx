/**
 * Onyx Web Dashboard Streaming Client
 * Connects to Onyx telemetry WebSocket / SSE streams and renders real-time throughput.
 */

export class OnyxStreamClient {
    constructor(host = "localhost", port = 9090) {
        this.host = host;
        this.port = port;
        this.connected = false;
        this.subscribers = [];
        this.reconnectInterval = 2000;
        this.pollTimer = null;
    }

    connect() {
        this.startPolling();
    }

    subscribe(callback) {
        this.subscribers.push(callback);
    }

    startPolling() {
        if (this.pollTimer) clearInterval(this.pollTimer);
        this.pollTimer = setInterval(async () => {
            try {
                const response = await fetch(`http://${this.host}:${this.port}/metrics`);
                if (response.ok) {
                    const data = await response.json();
                    this.connected = true;
                    this.notifySubscribers(data);
                } else {
                    this.connected = false;
                }
            } catch (err) {
                this.connected = false;
            }
        }, 1000);
    }

    notifySubscribers(data) {
        for (const cb of this.subscribers) {
            try {
                cb(data);
            } catch (e) {
                console.error("Subscriber callback error:", e);
            }
        }
    }

    disconnect() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = null;
        }
        this.connected = false;
    }
}

if (typeof window !== "undefined") {
    window.OnyxStreamClient = OnyxStreamClient;
}
