#!/usr/bin/env python3
"""Script to remove emojis from markdown files"""

import re
from pathlib import Path

# List of files to process
files = [
    "DOCUMENTATION-MASTER-INDEX.md",
    "README.md",
    "SCARCITY-COMPLETE-SYSTEM-DOCUMENTATION.md",
    "documentation/00-INDEX.md",
    "documentation/COMPLETE-SYSTEM-SUMMARY.md",
    "documentation/DOCUMENTATION-COMPLETE.md",
    "documentation/README-DOCUMENTATION.md",
    "DOCUMENTATION_INDEX.md",
    "COMPREHENSIVE_DOCUMENTATION.md",
    "SCARCITY-CORE-COMPLETE-REFERENCE.md",
]

# Emoji pattern - matches most common emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "]+",
    flags=re.UNICODE
)

def remove_emojis(text):
    """Remove emojis from text"""
    # Remove emojis
    text = emoji_pattern.sub('', text)
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text)
    # Clean up spaces at start of lines
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
    return text

def process_file(filepath):
    """Process a single file"""
    path = Path(filepath)
    if not path.exists():
        print(f"Skipping (not found): {filepath}")
        return
    
    print(f"Processing: {filepath}")
    try:
        # Read file
        content = path.read_text(encoding='utf-8')
        
        # Remove emojis
        cleaned = remove_emojis(content)
        
        # Write back
        path.write_text(cleaned, encoding='utf-8')
        print(f"Completed: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Main function"""
    print("Starting emoji removal...")
    for file in files:
        process_file(file)
    print("\nEmoji removal complete!")

if __name__ == "__main__":
    main()
