with open("prometheus/main.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(
    'signal_gen.generate_signal = _mock_gen',
    'signal_gen = _mock_gen'
)

with open("prometheus/main.py", "w", encoding="utf-8") as f:
    f.write(text)
