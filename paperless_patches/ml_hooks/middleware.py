"""ML-feedback floating-action-button middleware.

Injects a small floating button into Paperless-ngx's Angular SPA responses
so users can jump from any doc view to our /ml-ui/ feedback pages with one
click. The button is context-aware: on a /documents/<id>/ route it links
to /ml-ui/doc/<id>/; elsewhere it links to the /ml-ui/ index.

Why middleware, not a frontend fork:
    Paperless's frontend is Angular; modifying it requires forking the
    upstream repo, which we avoid (memory rule: updates from upstream
    compatible). Middleware injection into the base HTML response is the
    smallest change that adds the link without touching Angular source.

Attached at startup by ml_hooks.apps.MlHooksConfig._install_middleware().
Only fires on text/html responses with a closing </body>; static assets,
JSON API responses, streaming responses are all skipped.
"""
from __future__ import annotations

import logging

log = logging.getLogger("paperless.ml_hooks.middleware")


# Kept inline so the middleware has no additional static-file dependencies.
# Uses setInterval rather than pushState hooks because Angular routers do
# various things (hash routing, replaceState, etc.) and polling is simpler
# than chasing every variant.
_INJECT = b"""
<style id="ml-feedback-fab-style">
#ml-feedback-fab{
  position:fixed;bottom:20px;right:20px;
  padding:10px 16px;background:#17541f;color:#fff;
  border-radius:6px;text-decoration:none;
  font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  box-shadow:0 2px 8px rgba(0,0,0,.2);
  z-index:9999;
  transition:background .15s;
}
#ml-feedback-fab:hover{background:#1f6b28}
</style>
<script id="ml-feedback-fab-script">
(function () {
  if (document.getElementById('ml-feedback-fab')) return;
  var a = document.createElement('a');
  a.id = 'ml-feedback-fab';
  function update() {
    var m = location.pathname.match(/\\/documents\\/(\\d+)/);
    if (m) {
      a.href = '/ml-ui/doc/' + m[1] + '/';
      a.textContent = 'ML feedback: doc ' + m[1];
    } else {
      a.href = '/ml-ui/';
      a.textContent = 'ML feedback UI';
    }
  }
  update();
  // Paperless's Angular router changes location.pathname client-side
  // without reload. Poll once a second to keep the link in sync.
  setInterval(update, 1000);
  function mount() { document.body.appendChild(a); }
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
</script>
""".strip()


class MlFeedbackFabMiddleware:
    """Append the feedback FAB markup to HTML responses from paperless-web.

    Safe on responses that don't have a closing </body> (no-op). Updates
    Content-Length to match the new body size so proxies/browsers don't
    truncate the response.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Cheap filters first — skip anything that isn't a complete HTML
        # body. Streaming responses don't expose `.content`; skip via hasattr.
        content_type = response.get("Content-Type", "")
        if "text/html" not in content_type:
            return response
        if not hasattr(response, "content") or not response.content:
            return response
        if response.get("Content-Encoding"):
            # Response already compressed (gzip/br) — would need to decode,
            # inject, re-encode. Not worth the complexity; skip.
            return response

        body = response.content
        # Find the LAST </body> so embedded HTML in the document (e.g.,
        # doc previews) doesn't trip us. bytes.rfind returns -1 on no match.
        close_idx = body.rfind(b"</body>")
        if close_idx < 0:
            return response

        response.content = body[:close_idx] + _INJECT + body[close_idx:]
        response["Content-Length"] = str(len(response.content))
        return response
