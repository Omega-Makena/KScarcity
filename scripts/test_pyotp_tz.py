import pyotp
import datetime
import time

secret = pyotp.random_base32()
totp = pyotp.TOTP(secret)

tok_sys = totp.now()
tok_at_now = totp.at(datetime.datetime.now())
tok_at_utc = totp.at(datetime.datetime.utcnow())
tok_at_timestamp = totp.at(time.time())

print(f"totp.now(): {tok_sys}")
print(f"totp.at(now): {tok_at_now}")
print(f"totp.at(utcnow): {tok_at_utc}")
print(f"totp.at(time.time()): {tok_at_timestamp}")
