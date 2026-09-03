# diagram-gui — remote rendering: preview-then-handoff

How the viewer decides between local CPU rendering and handing a render off to a
configured `gemray-worker`, and how denoising and the TLS connection are handled
across that handoff. For everything else see [the README](../README.md); for the
`WorkerSettings` configuration itself see [settings.md](settings.md).

While the camera or light is moving, rendering is ordinary local CPU progressive
accumulation — nothing remote-specific happens. A repeating 100ms timer polls
the current camera/light pose; once it has been unchanged for a 600ms debounce
window *and* a remote worker is configured, the app hands off:

1. **The local preview buffer is discarded, never summed into the remote
   result.** A discard action always precedes sending the render request to the
   worker — there is no code path that carries a buffer from one source into
   the other. This mirrors `gemray-net`'s `FRAME`-vs-`PREVIEW` distinction:
   mixing a local partial accumulation into a remote one would be exactly the
   kind of un-composable mixing that protocol is built to prevent.
2. Local rendering is suspended entirely (a remote worker now owns the
   displayed image) and the remote worker starts streaming `FRAME`/`PREVIEW`
   deltas back, accumulated the normal `gemray-net` way.
3. If the user resumes dragging the camera mid-settle or mid-remote-render, the
   symmetric thing happens: a `CANCEL` is sent, the remote partial accumulation
   is discarded (never salvaged into the resumed local preview), and local
   previewing resumes from a clean buffer.

**Denoising is applied once, to the final merged accumulation buffer — never
per-frame and never per-source.** The À-Trous denoiser runs over the
accumulation's running *average*, never over the raw running sum (feeding
filtered output back into a progressive estimator would bias it), and it's a
single toggle covering the whole image: denoising is a nonlinear operation, so
running it separately on a local partial and a remote partial and then
combining the results would not equal running it once on the true combined
total. The same merge-then-denoise code path is reused for both local-only and
remote-sourced frames (remote `FRAME`/`PREVIEW` payloads carry only XYZ
radiance, never the depth/normal/facet-id guide buffers the denoiser also
needs, so those are regenerated locally with a cheap primary-ray-only prepass
before denoising a remote frame).

**The mutual-TLS connection is owned by one thread for its whole lifetime.**
TLS record state isn't safely readable and writable from two threads
concurrently the way a plaintext socket split would be, so
`bridge::remote_render` alternates, on one thread, between a short
timeout-bounded read attempt and a non-blocking check of an inbound command
channel — mirroring `gemray-worker`'s own emitter design — so a `CANCEL` can be
written promptly without a second thread ever touching the same connection.
