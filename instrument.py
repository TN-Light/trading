with open('prometheus/backtest/engine.py', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for i, l in enumerate(lines):
    if 'current_bar = data.iloc[i]' in l:
        lines.insert(i, '                if i % 100 == 0: print(f\'Processing bar {i}...\', flush=True)')
        break
with open('prometheus/backtest/engine.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
