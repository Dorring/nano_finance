def linearize_table(table_data):
    if not table_data or not isinstance(table_data, list):
        return ""
    header = table_data[0]
    lines = []
    for row in table_data[1:]:
        # 调试逻辑
        row_str = "; ".join([f"{h}: {v}" for h, v in zip(header, row) if v])
        lines.append(row_str)
    return " | ".join(lines)

table = [
    ["", "october 31 2009", "november 1 2008"],
    ["fair value", "$ 6427", "$ -23158"]
]
header = table[0]
row = table[1]
for h, v in zip(header, row):
    print(f"h:[{h}] v:[{v}] bool(v):{bool(v)}")

res = linearize_table(table)
print(f"Final Result: [{res}]")
