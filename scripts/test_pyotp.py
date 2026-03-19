import pyotp

secret = pyotp.random_base32()
print(f"Secret: {secret}")

totp = pyotp.TOTP(secret)
token = totp.now()
print(f"Token: {token}")

result = totp.verify(token)
print(f"Verify exact: {result}")

result2 = totp.verify(token + " ")
print(f"Verify with space: {result2}")
