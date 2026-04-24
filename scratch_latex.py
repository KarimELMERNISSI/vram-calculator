import re

with open(r"content\documentation.md", "r", encoding="utf-8") as f:
    text = f.read()

def latexify(eq):
    eq = eq.replace('×', '\\times ')
    eq = eq.replace('÷', '\\div ')
    eq = eq.replace('≈', '\\approx ')
    eq = eq.replace('−', '-')
    eq = eq.replace('≤', '\\leq ')
    eq = eq.replace('≥', '\\geq ')
    # For subscripts like V_total -> V_{total}
    eq = re.sub(r'([a-zA-Z]+)_([a-zA-Z0-9,]+)', r'\1_{\2}', eq)
    return eq

# Pattern 1
# > **Equation 1**  
# > `V_total = V_weights + V_kv + V_overhead`
pattern1 = re.compile(r'> \*\*Equation (\d+)\*\*\s*\n>\s*`([^`]+)`', re.MULTILINE)

def repl1(match):
    eq_num = match.group(1)
    eq_text = match.group(2)
    eq_text = latexify(eq_text)
    return f"$$ {eq_text} \\quad \\text{{(Equation {eq_num})}} $$"

text = pattern1.sub(repl1, text)

# Pattern 1b (sometimes there is text after the equation on the same line or next line)
# Handled well enough by pattern 1 if it just replaces the quote block.

# Pattern 2
# > `Latency per token = 1000 / TPS` ms &nbsp;&nbsp; *(Equation 32)*
pattern2 = re.compile(r'> `([^`]+)`\s*(.*?)\s*&nbsp;&nbsp;\s*[\*_]\(Equation (\d+)\)[\*_]', re.MULTILINE)

def repl2(match):
    eq_text = match.group(1)
    suffix = match.group(2).strip()
    eq_num = match.group(3)
    eq_text = latexify(eq_text)
    
    if suffix:
        return f"$$ {eq_text} \\quad \\text{{{suffix} (Equation {eq_num})}} $$"
    else:
        return f"$$ {eq_text} \\quad \\text{{(Equation {eq_num})}} $$"

text = pattern2.sub(repl2, text)

with open(r"content\documentation.md", "w", encoding="utf-8") as f:
    f.write(text)
print("Formulas replaced successfully.")
