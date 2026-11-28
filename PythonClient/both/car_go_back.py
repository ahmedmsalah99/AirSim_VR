import setup_path
import airsim
import time

# Connect to the car (same port and name you already use)
client = airsim.CarClient(port=41452)
client.confirmConnection()
client.enableApiControl(True, "Car1")

car_controls = airsim.CarControls()

print("Car1: Going backward")

# Reverse: a negative throttle in the reverse gear (in automatic gear it only brakes)
car_controls.throttle = -0.5
car_controls.steering = 0.0
car_controls.brake = 0.0
car_controls.handbrake = False
car_controls.is_manual_gear = True
car_controls.manual_gear = -1
car_controls.gear_immediate = True

for _ in range(80):   # ~4 seconds
    client.setCarControls(car_controls, "Car1")
    time.sleep(0.05)

# Stop the car
print("Car1: Stopping")
car_controls.throttle = 0.0
car_controls.brake = 1.0
car_controls.is_manual_gear = False   # back to automatic
car_controls.manual_gear = 0

for _ in range(20):
    client.setCarControls(car_controls, "Car1")
    time.sleep(0.05)

# Release control
client.enableApiControl(False, "Car1")
print("Done.")
