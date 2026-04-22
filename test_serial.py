import serial
import time
import argparse

PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"  # change this
BAUD = 115200

parser = argparse.ArgumentParser(description="Send relay delay messages to the controller.")
parser.add_argument("--port", default=PORT)
parser.add_argument("--baud", type=int, default=BAUD)
parser.add_argument("--crlf", action="store_true", help="Terminate messages with \\r\\n instead of \\n.")
args = parser.parse_args()

ser = serial.Serial(args.port, args.baud, timeout=1, write_timeout=1)

time.sleep(2)  # give ESP32 time to reset
ser.reset_input_buffer()

print("Type delay in ms (e.g. 2000). Ctrl+C to exit.")
print(f"Sending to {args.port} at {args.baud} baud.")

try:
    while True:
        delay = input("> ")

        if not delay:
            continue

        try:
            delay_ms = int(float(delay))
        except ValueError:
            print("Enter a number of milliseconds, e.g. 2000")
            continue

        line_ending = "\r\n" if args.crlf else "\n"
        message = f"{delay_ms}{line_ending}".encode("ascii")
        bytes_written = ser.write(message)
        ser.flush()
        print(f"sent {bytes_written} bytes: {message!r}")

        response = ser.readline()
        if response:
            print(f"received: {response!r}")

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    ser.close()
