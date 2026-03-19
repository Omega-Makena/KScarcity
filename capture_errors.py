from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def check_for_errors():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get("http://localhost:8501")
        
        # Wait for either the app to load normally, or an error box to appear
        time.sleep(5)
        
        # Streamlit errors are typically in elements with class 'stException' or 'element-container' containing Traceback
        error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Exception') or contains(text(), 'Traceback') or contains(text(), 'Error')]")
        
        if error_elements:
            print("--- VISIBLE ERRORS FOUND ON PAGE ---")
            for el in error_elements[:3]: # limit to top 3 to avoid spam
                text = el.text.strip()
                if len(text) > 10 and 'stException' not in text:
                    print(text)
                    print("-" * 40)
        else:
            print("No visible Exception/Traceback found on the main page.")
            
    except Exception as e:
        print(f"Failed to check page: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    check_for_errors()
