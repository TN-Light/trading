import yaml

with open("prometheus/config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

config["risk"]["drawdown_halt_pct"] = 50.0
config["risk"]["max_daily_loss_pct"] = 30.0
config["risk"]["max_daily_loss"] = 15000
config["risk"]["max_weekly_loss"] = 30000

with open("prometheus/config/settings.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
