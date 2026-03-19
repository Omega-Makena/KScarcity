import pyotp
import time
import datetime

secret = pyotp.random_base32()
print(f"Secret: {secret}")
totp = pyotp.TOTP(secret)

print(f"Time time(): {time.time()}")
print(f"Datetime now(): {datetime.datetime.now()}")
print(f"Datetime utcnow(): {datetime.datetime.utcnow()}")

print(f"PyOTP generated token: {totp.now()}")

uri = totp.provisioning_uri(name="test", issuer_name="test")
print(uri)
