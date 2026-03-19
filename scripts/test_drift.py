import pyotp
import datetime

secret = pyotp.random_base32()
totp = pyotp.TOTP(secret)

# The server is 10 minutes AHEAD of the phone.
# Phone time: T
# Server time: T + 10 mins

# Phone generates token for current time "T"
phone_time = datetime.datetime.now()
phone_code = totp.at(phone_time)

# Server verifies token against its own time "T + 10 mins"
server_time = phone_time + datetime.timedelta(minutes=10)
# But pyotp.verify() inherently uses datetime.datetime.now() inside, we can't mock its internal time easily.
# Wait, totp.verify() takes a `for_time` parameter? 
# Docs for pyotp: verify(self, otp, for_time=None, valid_window=0)
result = totp.verify(phone_code, for_time=server_time, valid_window=30)
print(f"Phone code: {phone_code}")
print(f"Server time: {server_time}")
print(f"Is valid with valid_window=30? {result}")

# What if window is too small?
print(f"Is valid with valid_window=5? {totp.verify(phone_code, for_time=server_time, valid_window=5)}")
print(f"Is valid with valid_window=20? {totp.verify(phone_code, for_time=server_time, valid_window=20)}")

