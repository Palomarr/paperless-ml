"""Force one scheduler tick with thresholds zeroed, for demo or manual retraining.

Invocation (from host):
    docker compose exec pipeline-scheduler python force_tick.py

The running scheduler is unaffected — this creates a transient
PipelineScheduler instance inside the same container with overridden
config (MIN_CORRECTIONS_DELTA=0, MIN_HOURS_SINCE_LAST_RUN=0) and runs
exactly one tick. Useful for demo narration ("here's what happens when
the conditions are met") without waiting for organic corrections +
time-floor to accumulate.

The real scheduler keeps its own state (last_trigger_time, correction
baseline) in its own process; this helper doesn't interfere with it.
"""
from scheduler import Config, PipelineScheduler


def main():
    cfg = Config.from_env()
    # Bypass both gate conditions
    cfg.min_corrections = 0
    cfg.min_hours = 0.0
    scheduler = PipelineScheduler(cfg)
    # Manually set a "baseline" of 0 so the corrections delta is always positive
    scheduler.last_correction_count = 0
    scheduler.last_trigger_time = 0.0
    scheduler.tick()


if __name__ == "__main__":
    main()
