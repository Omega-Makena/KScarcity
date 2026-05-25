import sys

def patch():
    with open('kshiked/ui/sentinel/router.py', 'r', encoding='utf-8') as f:
        text = f.read()

    target_nav = '''        "Document Intelligence": "DOCS",
        "Policy Intelligence": "POLICY_CHAT",
    }'''
    replacement_nav = '''        "Document Intelligence": "DOCS",
        "Policy Intelligence": "POLICY_CHAT",
        "Institution Portal": "INSTITUTION",
    }'''

    target_route = '''    elif view == "POLICY_CHAT":
        render_policy_chat(theme)'''
    replacement_route = '''    elif view == "POLICY_CHAT":
        render_policy_chat(theme)
    elif view == "INSTITUTION":
        from institution.page import render as render_institution
        render_institution(theme)'''

    if target_nav in text and target_route in text:
        text = text.replace(target_nav, replacement_nav)
        text = text.replace(target_route, replacement_route)
        with open('kshiked/ui/sentinel/router.py', 'w', encoding='utf-8') as f:
            f.write(text)
        print('Successfully patched router.py')
    else:
        print('Failed to find exact text blocks for replacement')
        sys.exit(1)

if __name__ == '__main__':
    patch()
