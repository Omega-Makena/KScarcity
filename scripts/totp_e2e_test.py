import asyncio
from playwright.async_api import async_playwright
import pyotp

async def run():
    async with async_playwright() as p:
        # Launch browser headless
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Connect to Streamlit on port 8501
        print("Navigating to http://localhost:8501")
        await page.goto("http://localhost:8501")
        
        # Wait for Streamlit to load completely
        await page.wait_for_selector("text=Enter Secure Portal", timeout=10000)

        # 1. Enter Secure Portal
        print("Clicking Enter Secure Portal")
        await page.locator("text=Enter Secure Portal").click()
        await page.wait_for_timeout(2000) # Give it time to render phase 1

        # 2. Phase 1 Login
        print("Filling Username")
        # Find the text input inside the form
        inputs = await page.locator("input[type='text']").all()
        await inputs[0].fill("admin")
        await page.locator("text=Authenticate").click()
        
        # Wait for Phase 2 to render
        print("Waiting for Phase 2 Verification...")
        await page.wait_for_timeout(3000)

        # 3. Read the secret from the DEBUG banner
        debug_texts = await page.locator("text=DEBUGGING SECRET TRACE:").all_text_contents()
        if not debug_texts:
            print("ERROR: Could not find debug banner!")
            await browser.close()
            return
            
        debug_text = debug_texts[0]
        # "DEBUGGING SECRET TRACE: The active verification secret for this session is `ABCDEF...`."
        secret = debug_text.split("`")[1]
        print(f"Extracted Secret from Streamlit: {secret}")
        
        # Generate the matching TOTP
        totp = pyotp.TOTP(secret)
        current_code = totp.now()
        print(f"Generated TOTP code: {current_code}")
        
        # 4. Input the code into Phase 2
        print("Submitting 2FA Setup form...")
        inputs = await page.locator("input[type='text']").all()
        # the setup_token is the first/only text input
        await inputs[0].fill(current_code)
        
        await page.locator("text=Enable 2FA & Enter").click()
        
        # Wait for success
        await page.wait_for_timeout(2000)
        
        # Check if success or failure
        success_texts = await page.locator("text=2FA Successfully Enabled!").all_text_contents()
        if success_texts:
            print("SUCCESS! 2FA fully enabled!")
        else:
            errors = await page.locator("text=Invalid code").all_text_contents()
            if errors:
                print("FAILED! Streamlit rejected the valid code with: 'Invalid code'")
            else:
                print("UNKNOWN STATUS. Here's all text on the page:")
                print(await page.locator("body").inner_text())
                
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
