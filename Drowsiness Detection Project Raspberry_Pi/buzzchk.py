import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
buzzer_pin = 17  # Replace with the actual GPIO pin you connect the buzzer to
GPIO.setup(buzzer_pin, GPIO.OUT)

def beep_once():
    GPIO.output(buzzer_pin, GPIO.HIGH)  # Turn on the buzzer
    time.sleep(1)
    GPIO.output(buzzer_pin, GPIO.LOW)  # Turn off the buzzer

# Example usage
beep_once()

# Cleanup GPIO
GPIO.cleanup()
