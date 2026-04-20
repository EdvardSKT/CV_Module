import serial
import time

PORT = "/dev/cu.usbserial-210"  # change this
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)  # give ESP32 time to reset

print("Type delay in ms (e.g. 2000). Ctrl+C to exit.")

try:
    while True:
        delay = input("> ")

        if not delay:
            continue

        ser.write((delay + "\n").encode())

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    ser.close()