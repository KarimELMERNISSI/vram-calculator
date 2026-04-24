import re

with open(r"content\documentation.md", "r", encoding="utf-8") as f:
    text = f.read()

def fix_latex(text):
    # Fix subscript grouping inside $$ ... $$ blocks
    def fix_match(m):
        eq = m.group(1)
        
        # fix simple unbraced subscripts that have multiple letters (like _total, _VRAM, _RAM, _PCIe, _kv)
        # We need to find `_X` where X is more than one letter and brace it `_{X}`
        # However, it's safer to just brace any sequence of word characters after `_`
        # But be careful not to double brace `_{...}`
        
        # First temporarily unbrace any already braced ones to make it uniform
        eq = re.sub(r'_\{([^\}]+)\}', r'_\1', eq)
        
        # Now brace everything after `_` that consists of word characters
        # But wait, there might be spaces or other things.
        # Actually it's easier to just match `_` followed by alphanumeric/commas and brace it
        eq = re.sub(r'_([a-zA-Z0-9,]+)', r'_{\1}', eq)
        
        # specific fixes for math
        eq = eq.replace('min(', '\\min(')
        eq = eq.replace('max(', '\\max(')
        
        return f"$$ {eq} $$"

    text = re.sub(r'\$\$(.*?)\$\$', fix_match, text, flags=re.DOTALL)
    return text

text = fix_latex(text)

with open(r"content\documentation.md", "w", encoding="utf-8") as f:
    f.write(text)
print("Latex formatting fixed.")
