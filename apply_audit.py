import os
import re

files = ["content/documentation.md", "docs/md/vram_calc_doc.md"]

def apply_fixes(content):
    # 1. TFLOPS
    content = content.replace("990 TFLOPS", "1,979 TFLOPS (with sparsity)")
    content = content.replace("990/3350", "1979/3350")
    content = content.replace("295 FLOP/byte", "591 FLOP/byte")
    
    # 2. BW formula (using regex just in case it differs slightly)
    content = re.sub(
        r'3,?930\s*×\s*5,?120\s*/\s*8\s*=\s*2,?515\s*GB/s',
        r'6,864 × 5,120 / 8 / 1000 ≈ 4,392 GB/s (theoretical raw per stack; actually 3,350 GB/s total)',
        content
    )
    
    # 3. MI300X memory type - we already did this in gpus.json, but let's check table
    # The user noted documentation.md Section 10 is missing the HBM type column.
    
    # 4. Chiplets
    content = content.replace("8 chiplets (8 compute + 4 I/O)", "12 chiplets (8 XCDs + 4 I/O dies)")
    content = content.replace("8 chiplets (8 XCDs + 4 I/O dies)", "12 chiplets (8 XCDs + 4 I/O dies)")
    
    # 5. H200 HBM stacks
    content = content.replace("H200 (4 stacks, 4,800 GB/s)", "H200 (6 stacks, 4,800 GB/s)")
    content = content.replace("1,229 GB/s", "~800 GB/s")
    
    # 6. PCIe 3.0+
    content = content.replace("128/130", "128/130 (for PCIe Gen 3.0+)")
    content = content.replace("128/130 (for PCIe Gen 3.0+) (for PCIe Gen 3.0+)", "128/130 (for PCIe Gen 3.0+)")
    
    # 7. GGUF
    content = content.replace("GPT-Generated Unified Format", "GGML GPT-Generated Unified Format")
    
    # 8. Tensor Core vs CUDA cores
    content = content.replace("(tensor cores)", "(CUDA cores, FMA operation)")
    
    # Ensure H100 and H200 in any table or text have 1979 instead of 990 (not covered by above string replace)
    content = re.sub(r'\|\s*H200 SXM\s*\|\s*141\s*\|\s*4800\s*\|\s*990\s*\|', '| H200 SXM | 141 | 4800 | 1979 |', content)
    content = re.sub(r'\|\s*H100 SXM\s*\|\s*80\s*\|\s*3350\s*\|\s*990\s*\|', '| H100 SXM | 80 | 3350 | 1979 |', content)
    
    return content

for path in files:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = apply_fixes(content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed {path}")
    else:
        print(f"Not found: {path}")

