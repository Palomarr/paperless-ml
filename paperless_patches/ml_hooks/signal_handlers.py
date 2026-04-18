import logging

from ml_hooks import tasks

log = logging.getLogger("ml_hooks")


def on_consumption_finished(sender, document, **kwargs):
    log.info("ml_hooks: consumption finished for doc %s", document.pk)
    tasks.htr_transcribe.delay(document.pk)
    tasks.encode_document.delay(document.pk)


def on_document_updated(sender, document, **kwargs):
    log.info("ml_hooks: document updated %s -> re-encode", document.pk)
    tasks.encode_document.delay(document.pk)
