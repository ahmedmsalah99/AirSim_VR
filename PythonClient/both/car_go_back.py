import setup_path
import airsim
import time

# Connect to the car (same port and name you already use)
client = airsim.CarClient(port=41452)
client.confirmConnection()
client.enableApiControl(True, "Car1")

car_controls = airsim.CarControls()

print("Car1: Going backward")

# === OPTION 1: Reverse using negative throttle (automatic gear) ===
car_controls.throttle = -0.5      # negative = reverse
car_controls.steering = 0.0
car_controls.brake = 0.0
car_controls.handbrake = False
car_controls.is_manual_gear = False

for _ in range(80):   # ~4 seconds
    client.setCarControls(car_controls, "Car1")
    time.sleep(0.05)

# Stop the car
print("Car1: Stopping")
car_controls.throttle = 0.0
car_controls.brake = 1.0

for _ in range(20):
    client.setCarControls(car_controls, "Car1")
    time.sleep(0.05)

# Release control
client.enableApiControl(False, "Car1")
print("Done.")
