import re
with open("prometheus/main.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add apex argument to parser
if "--apex" not in text:
    text = text.replace(
        "parser_backtest.add_argument('--intraday', action='store_true'",
        "parser_backtest.add_argument('--apex', action='store_true')\n    parser_backtest.add_argument('--intraday', action='store_true'"
    )
    text = text.replace(
        'parser_backtest.add_argument("--intraday", action="store_true"',
        'parser_backtest.add_argument("--apex", action="store_true")\n    parser_backtest.add_argument("--intraday", action="store_true"'
    )

# Pass apex args to run_intraday_backtest
text = text.replace("system.run_intraday_backtest(days=args.days, symbol=args.symbol)", "system.run_intraday_backtest(days=args.days, symbol=args.symbol, apex=getattr(args, 'apex', False))")

# Instantiate apex in _run_intraday_backtest_on_slice
apex_logic = """
        if apex:
            from prometheus.signals.apex_generator import ApexGenerator
            from prometheus.signals.aes_fusion import AESFusionEngine
            from prometheus.execution.lap_recovery import LAPRecoveryManager
            
            aes = AESFusionEngine()
            apex_gen = ApexGenerator(aes)
            # monkey patch
            def generate_signal(*args, **kwargs):
                return apex_gen.generate_signal(df_slice, i)
            engine.signal_generator.generate_signal = generate_signal
"""

if "if apex:" not in text:
    target = "for i in range(len(df_slice) - 1):"
    text = text.replace(target, apex_logic + "\n        " + target)

with open("prometheus/main.py", "w", encoding="utf-8") as f:
    f.write(text)
