import re

def fix_math(match):
    math_str = match.group(0)
    # Fix subscripts
    math_str = math_str.replace('*{', '_{')
    math_str = math_str.replace('\\_{', '_{')
    math_str = math_str.replace(',}', '}')
    math_str = math_str.replace('*B', '_B')
    math_str = math_str.replace('*C', '_C')
    math_str = math_str.replace('*D', '_D')
    math_str = math_str.replace('CO₂,hour', 'CO_{2,hour}')
    math_str = math_str.replace('CO₂,annual', 'CO_{2,annual}')
    
    # Fix instances of unbraced asterisks
    math_str = math_str.replace('TPS*eff', 'TPS_{eff}')
    math_str = math_str.replace('U*active', 'U_{active}')
    math_str = math_str.replace('U*swapped', 'U_{swapped}')
    
    return math_str

def renumber_eq(match):
    prefix = match.group(1)
    num = int(match.group(2))
    suffix = match.group(3)
    new_num = num + 12
    return f"{prefix}{new_num}{suffix}"

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix math blocks
    content = re.sub(r'\$\$.*?\$\$', fix_math, content, flags=re.DOTALL)

    # Renumber equations in Part 2
    separator = '## 4. Connectivity & Bandwidth Architecture'
    if separator in content:
        parts = content.split(separator)
        part1 = parts[0]
        part2 = parts[1]
        
        # Replace (Equation X)
        part2 = re.sub(r'(\(Equation )(\d+)(\))', renumber_eq, part2)
        # Replace **Equation X**
        part2 = re.sub(r'(\*\*Equation )(\d+)(\*\*)', renumber_eq, part2)
        
        content = part1 + separator + part2

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    process_file('content/documentation.md')
