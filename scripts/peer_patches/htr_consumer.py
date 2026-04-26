"""
Long-lived Kafka consumer for Phase 2 — PATCHED for ongoing-operation stability.

This is a drop-in replacement for paperless_data_integration/htr_consumer/consumer.py
mounted into the container via docker-compose volume override (see
scripts/up_peer_components.sh up_htr_consumer override block).

Why this patch exists:

  Elnath's upstream consumer.py uses kafka-python's defaults for
  max_poll_interval_ms (300s = 5 min) and max_poll_records (500). Our
  R6 sample-trace test uploaded a 241-page handwritten book. Slicer +
  HTR processing took 412s end-to-end — exceeding max_poll_interval_ms,
  causing Kafka to declare the consumer dead, rebalance the partition,
  and reject our commit:

      kafka.errors.CommitFailedError: CommitFailedError: Commit cannot
      be completed since the group has already rebalanced and assigned
      the partitions to another member.

  On restart, the consumer reads from the last *committed* offset
  (before the long doc), reprocesses the same long doc, hits the same
  timeout — infinite loop.

  The two patched lines below add env-var-driven kwargs that the
  upstream consumer doesn't expose:

    max_poll_interval_ms = KAFKA_MAX_POLL_INTERVAL_MS env var
                           (default 30 min — handles books, manuals)
    max_poll_records     = KAFKA_MAX_POLL_RECORDS env var
                           (default 1 — finish one doc fully before next poll)

Long-term: submit upstream PR to expose these env vars in Elnath's repo.
For now this is the workaround that keeps htr_consumer stable on long
documents during ongoing-operation (April 28-May 4).
"""

import json
import logging
import os
import signal
import sys
import time

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from slicer import RegionSlicer
import processor


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("htr_consumer")


def _build_consumer() -> KafkaConsumer:
    broker = os.environ.get("KAFKA_BROKER", "redpanda:9092")
    topic  = os.environ.get("KAFKA_TOPIC", "paperless.uploads")
    group  = os.environ.get("KAFKA_GROUP_ID", "htr-preprocessing")

    # PATCH: env-driven kafka tuning to prevent rebalance-on-long-document.
    # Upstream Elnath's consumer.py uses defaults (300s poll interval, 500 records).
    # Our R6 sample-trace produced 412s processing on a 241-page book; default
    # times out and infinite-loops. These overrides keep the consumer stable.
    max_poll_interval_ms = int(os.environ.get("KAFKA_MAX_POLL_INTERVAL_MS", "1800000"))  # 30 min
    max_poll_records     = int(os.environ.get("KAFKA_MAX_POLL_RECORDS", "1"))            # one at a time

    while True:
        try:
            c = KafkaConsumer(
                topic,
                bootstrap_servers=broker,
                group_id=group,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                max_poll_interval_ms=max_poll_interval_ms,
                max_poll_records=max_poll_records,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
            log.info(
                "Connected to Kafka at %s, topic=%s, group=%s "
                "(max_poll_interval_ms=%d, max_poll_records=%d)",
                broker, topic, group, max_poll_interval_ms, max_poll_records,
            )
            return c
        except NoBrokersAvailable:
            log.warning("Kafka not ready at %s, retrying in 5s", broker)
            time.sleep(5)


def _build_slicer() -> RegionSlicer:
    return RegionSlicer(
        paperless_url   = os.environ.get("PAPERLESS_URL", "http://paperless-webserver-1:8000"),
        paperless_token = os.environ.get("PAPERLESS_TOKEN", ""),
        minio_endpoint  = os.environ.get("MINIO_ENDPOINT", "minio:9000"),
        minio_access_key= os.environ.get("MINIO_ACCESS_KEY", "admin"),
        minio_secret_key= os.environ.get("MINIO_SECRET_KEY", "paperless_minio"),
        minio_bucket    = os.environ.get("MINIO_BUCKET", "paperless-images"),
    )


def main() -> None:
    stop = False
    def _handle_signal(signum, _frame):
        nonlocal stop
        log.info("Received signal %d, stopping after current event", signum)
        stop = True
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    if not os.environ.get("PAPERLESS_TOKEN"):
        log.error("PAPERLESS_TOKEN env var is required (needed for slicer to fetch documents).")
        sys.exit(1)

    slicer = _build_slicer()
    consumer = _build_consumer()

    log.info("HTR preprocessing consumer ready. Waiting for events...")

    for msg in consumer:
        if stop:
            break
        event = msg.value
        log.info(
            "recv offset=%d partition=%d paperless_doc_id=%s",
            msg.offset, msg.partition, event.get("paperless_doc_id"),
        )
        try:
            processor.process_event(event, slicer)
        except Exception as exc:
            log.exception("Failed to process event offset=%d: %s", msg.offset, exc)
        finally:
            consumer.commit()

    consumer.close()
    log.info("Consumer exited cleanly.")


if __name__ == "__main__":
    main()
