import paho.mqtt.client as mqtt
import json
import time
import threading
import socket

class MQTTClientWrapper:
    def __init__(self, client_id, broker_address="localhost", broker_port=1883):
        self.client_id = client_id
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False
        
        self.callbacks = {} # topic -> function
        self.message_buffer = {} # topic -> latest_message
        self.lock = threading.Lock()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[{self.client_id}] Connected to MQTT Broker")
            self.connected = True
            # Resubscribe if we had subscriptions? 
            # For now, simplistic re-sub logic handled by user calling subscribe again or just standard persistent session if configured.
            # But we'll just let the user code handle verify.
        else:
            print(f"[{self.client_id}] Failed to connect, return code {rc}")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except:
            payload = msg.payload.decode()
            
        with self.lock:
            self.message_buffer[topic] = payload
            
        # Trigger specific callback if exists
        # Handle wildcards rudimentarily or exact match
        for sub_topic, callback in self.callbacks.items():
            if mqtt.topic_matches_sub(sub_topic, topic):
                callback(topic, payload)

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.broker_port, 60)
            self.client.loop_start()
            # Wait for connection
            start = time.time()
            while not self.connected and time.time() - start < 2.0:
                time.sleep(0.1)
        except Exception as e:
            print(f"[{self.client_id}] Connection Error: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def subscribe(self, topic, callback=None):
        self.client.subscribe(topic)
        if callback:
            self.callbacks[topic] = callback

    def publish(self, topic, message):
        if not self.connected:
            return
        
        if isinstance(message, dict) or isinstance(message, list):
            payload = json.dumps(message)
        else:
            payload = str(message)
            
        self.client.publish(topic, payload)

    def get_last_message(self, topic):
        with self.lock:
            return self.message_buffer.get(topic)
