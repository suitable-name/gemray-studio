use crate::serve::{
    connection::{BUILD_MISMATCH_CODE, VALIDATION_FAILED_CODE},
    handle_connection,
};
use diagram_catalog::db::sqlite::Database;
use gemray::{
    geometry::cuts::StandardGemCuts,
    optics::{materials::GemMaterial, raytracer::LightingPreset},
};
use gemray_net::{
    SceneState, handshake,
    messages::{
        Backend, Cancel, ClientMessage, Done, ErrorMsg, Hello, RenderRequest, StreamEvent, Welcome,
    },
};
use std::{
    io::{Cursor, Read, Write},
    net::TcpListener,
    thread,
};

use crate::{render_core, stream_emit::TimeoutRead};

/// A fresh, empty, throwaway temp database for tests in this module that only need
/// SOME `&Database` to pass to `handle_connection` -- none of these tests exercise the
/// library protocol itself (see `crate::serve::library`'s own tests for that); they
/// only need a valid handle to satisfy the connection handler's signature. Per this
/// crate's own hard rule: tests never touch `facet_diagrams.sqlite`, only their own
/// throwaway temp files.
fn test_db() -> Database {
    let path = std::env::temp_dir().join(format!(
        "gemray-worker-serve-test-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    Database::new(Some(path.to_str().unwrap())).unwrap()
}

fn tiny_scene() -> SceneState {
    SceneState {
        width: 4,
        height: 4,
        yaw: 0.4,
        pitch: 0.3,
        distance: 3.0,
        light_yaw: 0.85,
        light_pitch: 0.95,
        exposure: 1.0,
        max_bounces: 4,
        lighting_preset: LightingPreset::Daylight,
        material: GemMaterial::diamond(),
        planes: StandardGemCuts::standard_round_brilliant(),
        girdle_frosted: false,
    }
}

/// A `Read + Write` over two independent in-memory buffers, standing in for one
/// end of a duplex connection: writes go to `out`, reads come from `in_`. Lets
/// `handle_connection` be driven directly with hand-assembled request bytes,
/// without any actual networking.
///
/// `TimeoutRead`-aware: mirrors a real socket's read-timeout behavior closely
/// enough to exercise `stream_emit::poll_for_client_message`'s polling loop without any
/// actual networking. A real blocking socket with NO timeout set just blocks
/// forever on an idle connection, which an in-memory `Cursor` can't do -- so with
/// no timeout set (`timeout_active: false`, the default every existing test relies
/// on unchanged), exhausting `in_` reports `Ok(0)` (real EOF), exactly like before
/// this type learned about timeouts at all. With a short timeout SET (as
/// `run_stream`'s emitter does for the duration of a request's streaming phase),
/// exhausting `in_` instead reports `WouldBlock` -- "nothing new since last time",
/// not "the peer hung up" -- letting a test simulate "connection still open, no
/// `CANCEL` sent (yet)" by simply not writing one into `in_` at all.
struct DuplexHalf {
    in_: Cursor<Vec<u8>>,
    out: Vec<u8>,
    timeout_active: bool,
}

impl DuplexHalf {
    const fn new(input: Vec<u8>) -> Self {
        Self {
            in_: Cursor::new(input),
            out: Vec::new(),
            timeout_active: false,
        }
    }
}

impl Read for DuplexHalf {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.in_.read(buf)?;
        if n == 0 && self.timeout_active {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "DuplexHalf: no more scripted input (yet)",
            ));
        }
        Ok(n)
    }
}

impl Write for DuplexHalf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.out.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl TimeoutRead for DuplexHalf {
    fn set_read_timeout(&mut self, duration: Option<std::time::Duration>) -> std::io::Result<()> {
        self.timeout_active = duration.is_some();
        Ok(())
    }
}

const fn live_progressive(cadence_ms: u32) -> gemray_net::messages::StreamConfig {
    gemray_net::messages::StreamConfig {
        transfer_mode: gemray_net::messages::TransferMode::LiveProgressive,
        cadence_ms,
        preview: None,
    }
}

const fn final_only(cadence_ms: u32) -> gemray_net::messages::StreamConfig {
    gemray_net::messages::StreamConfig {
        transfer_mode: gemray_net::messages::TransferMode::FinalOnly,
        cadence_ms,
        preview: None,
    }
}

/// Reads [`gemray_net::messages::StreamEvent`]s from `reader` (pairing each with
/// its raw payload, for `Frame`/`Preview`) until -- and including -- a `Done` or
/// `Error`, the two terminal variants for one `RENDER` reply.
fn read_stream_until_done<R: Read>(
    reader: &mut R,
) -> Vec<(gemray_net::messages::StreamEvent, Option<Vec<u8>>)> {
    let mut events = Vec::new();
    loop {
        let (event, payload) = gemray_net::messages::read_stream_event(reader).unwrap();
        let terminal = matches!(
            event,
            gemray_net::messages::StreamEvent::Done(_)
                | gemray_net::messages::StreamEvent::Error(_)
        );
        events.push((event, payload));
        if terminal {
            break;
        }
    }
    events
}

#[test]
fn handle_connection_refuses_a_mismatched_build_hash() {
    let mut input = Vec::new();
    let bad_hello = Hello {
        protocol_version: gemray_net::messages::PROTOCOL_VERSION,
        build_hash: [0xAB; 8],
    };
    gemray_net::messages::write_message(&mut input, &bad_hello).unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 1, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let err: ErrorMsg = gemray_net::messages::read_message(&mut out_cursor).unwrap();
    assert_eq!(err.code, BUILD_MISMATCH_CODE);
    assert!(err.message.contains("refusing to pair"), "{}", err.message);
}

#[test]
fn handle_connection_accepts_a_matching_build_hash_and_sends_welcome() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();
    // No RenderRequest follows -- the reader hits EOF, which handle_connection
    // treats as the peer having closed the connection after the handshake.

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 1, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();
    assert_eq!(welcome.build_hash, handshake::local_hello().build_hash);
    assert!(matches!(
        welcome.render.as_ref().unwrap().backend,
        Backend::Cpu { .. }
    ));
}

#[test]
fn handle_connection_rejects_a_request_that_fails_validation_but_keeps_the_connection_open() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();

    let mut bad_scene = tiny_scene();
    bad_scene.planes[0].normal = [f32::NAN, 0.0, 0.0];
    let bad_request = RenderRequest {
        request_id: 1,
        scene: bad_scene,
        first_sample: 0,
        samples: 4,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(bad_request)),
    )
    .unwrap();

    // A second, well-formed request right after the bad one -- proves the
    // connection is still alive and serving requests after a validation failure.
    let good_request = RenderRequest {
        request_id: 2,
        scene: tiny_scene(),
        first_sample: 0,
        samples: 2,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(good_request)),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 1, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();

    // The bad request's reply: a single StreamEvent::Error, nothing else.
    let bad_events = read_stream_until_done(&mut out_cursor);
    assert_eq!(bad_events.len(), 1);
    let StreamEvent::Error(err) = &bad_events[0].0 else {
        panic!("expected StreamEvent::Error, got {:?}", bad_events[0].0);
    };
    assert_eq!(err.code, VALIDATION_FAILED_CODE);

    // The good request's reply: FinalOnly, so exactly one Frame (with correct
    // request_id and sample count) followed by Done { cancelled: false }.
    let good_events = read_stream_until_done(&mut out_cursor);
    let frames: Vec<_> = good_events
        .iter()
        .filter_map(|(e, p)| match e {
            StreamEvent::Frame(h) => Some((h, p.as_ref().unwrap())),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 1);
    let (header, payload) = frames[0];
    assert_eq!(header.request_id, 2);
    assert_eq!(header.samples, 2);
    assert_eq!(payload.len(), 4 * 4 * gemray_net::radiance::BYTES_PER_PIXEL);
    assert!(matches!(
        good_events.last().unwrap().0,
        StreamEvent::Done(Done {
            cancelled: false,
            ..
        })
    ));
}

/// The test that proves the whole path: a real `RenderRequest` for a sample
/// range, over a real loopback `TcpStream`, driven through `handle_connection`
/// directly (not `run`'s accept loop, so the test doesn't need to manage a
/// long-running listener thread) -- request goes in, a correctly-sized summed
/// radiance buffer comes back.
#[test]
fn serve_round_trip_over_a_loopback_socket_returns_a_correctly_sized_buffer() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        handle_connection(stream, 2, &test_db()).unwrap();
    });

    let mut client = std::net::TcpStream::connect(addr).unwrap();
    gemray_net::messages::write_message(&mut client, &handshake::local_hello()).unwrap();
    let welcome: Welcome = gemray_net::messages::read_message(&mut client).unwrap();
    assert!(matches!(
        welcome.render.as_ref().unwrap().backend,
        Backend::Cpu { .. }
    ));

    let scene = tiny_scene();
    let request = RenderRequest {
        request_id: 5,
        scene,
        first_sample: 10,
        samples: 3,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut client,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();

    let events = read_stream_until_done(&mut client);
    let frames: Vec<_> = events
        .iter()
        .filter_map(|(e, p)| match e {
            StreamEvent::Frame(h) => Some((h, p.as_ref().unwrap())),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 1);
    let (header, payload) = frames[0];
    assert_eq!(header.request_id, 5);
    assert_eq!(header.first_sample, 10);
    assert_eq!(header.samples, 3);
    assert_eq!(payload.len(), 4 * 4 * gemray_net::radiance::BYTES_PER_PIXEL);

    drop(client); // close the connection so handle_connection's read loop sees EOF and returns
    server.join().unwrap();
}

#[test]
fn serve_round_trip_result_matches_tracing_the_same_range_directly() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let scene = tiny_scene();
    let scene_for_server = scene.clone();

    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        handle_connection(stream, 2, &test_db()).unwrap();
    });

    let mut client = std::net::TcpStream::connect(addr).unwrap();
    gemray_net::messages::write_message(&mut client, &handshake::local_hello()).unwrap();
    let _welcome: Welcome = gemray_net::messages::read_message(&mut client).unwrap();

    let request = RenderRequest {
        request_id: 6,
        scene: scene.clone(),
        first_sample: 0,
        samples: 4,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut client,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();
    let events = read_stream_until_done(&mut client);
    let (_header, payload) = events
        .iter()
        .find_map(|(e, p)| match e {
            StreamEvent::Frame(h) => Some((h, p.as_ref().unwrap())),
            _ => None,
        })
        .unwrap();
    drop(client);
    server.join().unwrap();

    let over_the_wire = gemray_net::radiance::decode(payload, scene.width, scene.height).unwrap();
    let direct = render_core::trace_samples(&scene_for_server, 0, 4, 2);
    // Relative tolerance, not bit-exact -- the server now sub-batches the request
    // internally (see `stream_emit::run_tracer`'s adaptive sizing) and sums the
    // sub-batches together, rather than tracing the whole range in one call the way
    // `direct` does here; float addition isn't associative, so a different grouping
    // can differ in the last bit or two even when both are correct. See
    // `render_core::tests::splitting_a_sample_range_across_two_calls_sums_to_the_same_result_as_one_call`,
    // whose same tolerance this mirrors.
    for (a, b) in over_the_wire.iter().zip(&direct) {
        let diff = (*a - *b).abs();
        let scale = a.abs().max(b.abs()).max(glam::Vec3::splat(1e-6));
        assert!(
            (diff / scale).max_element() < 1e-3,
            "over_the_wire={a:?} direct={b:?}"
        );
    }
}

/// The GPU counterpart to `serve_round_trip_over_a_loopback_socket_returns_a_correctly_sized_buffer`:
/// acquires a real [`gemray::renderer::gpu_backend::GpuBackend`] (rather than
/// `GpuBackend::disabled`, what `handle_connection`'s CPU-only wrapper always uses) and
/// drives `handle_connection_with_gpu` directly, over a real loopback socket. On a
/// machine with a usable adapter this proves `WELCOME` reports `Backend::Gpu` with a real
/// adapter label AND that the request that follows still comes back with a
/// correctly sized radiance buffer -- exercising the GPU dispatch end to end, not
/// merely compiling it. On a machine with no usable adapter, `GpuBackend::acquire`
/// itself falls back to declining (see its own doc comment), so this degrades to
/// re-checking the CPU behavior rather than failing -- this crate's CI/dev machine
/// has a working adapter (see the workspace root instructions), so this is expected
/// to genuinely exercise the GPU path there.
#[cfg(feature = "gpu")]
#[test]
fn serve_round_trip_over_gpu_reports_backend_gpu_and_a_correctly_sized_buffer() {
    use crate::serve::handle_connection_with_gpu;
    use gemray::renderer::gpu_backend::GpuBackend;
    use std::{net::TcpStream, sync::Arc};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let scene = tiny_scene(); // diamond -- isotropic, so GpuBackend accepts it.

    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let gpu = Arc::new(GpuBackend::acquire());
        handle_connection_with_gpu(stream, 2, &gpu, &test_db()).unwrap();
    });

    let mut client = TcpStream::connect(addr).unwrap();
    gemray_net::messages::write_message(&mut client, &handshake::local_hello()).unwrap();
    let welcome: Welcome = gemray_net::messages::read_message(&mut client).unwrap();

    let request = RenderRequest {
        request_id: 7,
        scene: scene.clone(),
        first_sample: 0,
        samples: 4,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut client,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();
    let events = read_stream_until_done(&mut client);
    let (header, payload) = events
        .iter()
        .find_map(|(e, p)| match e {
            StreamEvent::Frame(h) => Some((h, p.as_ref().unwrap())),
            _ => None,
        })
        .unwrap();
    assert_eq!(header.samples, 4);
    assert_eq!(
        payload.len(),
        scene.width as usize * scene.height as usize * gemray_net::radiance::BYTES_PER_PIXEL
    );

    drop(client);
    server.join().unwrap();

    // Only asserted if this machine actually has a usable adapter (see this test's
    // own doc comment) -- a machine with none legitimately reports Backend::Cpu,
    // which is exactly the "never claim GPU while silently tracing on CPU" contract
    // this crate exists to uphold, not a test failure.
    if let Some(Backend::Gpu { adapter }) = welcome.render.map(|r| r.backend) {
        assert!(!adapter.is_empty(), "adapter label must be non-empty");
    }
}

// ---- Progressive streaming ---------------------------------------------------

/// A scene with enough per-sample work (real gem geometry, several bounces) that a
/// several-dozen-sample request takes long enough for `run_stream`'s emitter to get
/// multiple chances to poll/emit before the tracer finishes -- unlike `tiny_scene`,
/// which a handful of samples can finish tracing before the emitter's very first
/// loop iteration even runs. Used by tests that need to observe more than one
/// emission (tiling, `PROGRESS` under `FinalOnly`) deterministically rather than by
/// luck of thread scheduling.
fn heavier_scene() -> SceneState {
    SceneState {
        width: 24,
        height: 24,
        yaw: 0.4,
        pitch: 0.3,
        distance: 3.0,
        light_yaw: 0.85,
        light_pitch: 0.95,
        exposure: 1.0,
        max_bounces: 6,
        lighting_preset: LightingPreset::Daylight,
        material: GemMaterial::diamond(),
        planes: StandardGemCuts::standard_round_brilliant(),
        girdle_frosted: false,
    }
}

/// The `request_id` carried by one `StreamEvent` -- every variant but `Error`
/// carries one; see `gemray_net::messages`' docs on why that's what makes a stale
/// reply mechanically identifiable.
fn event_request_id(event: &StreamEvent) -> u32 {
    match event {
        StreamEvent::Frame(h) => h.request_id,
        StreamEvent::Preview(h) => h.request_id,
        StreamEvent::Progress(p) => p.request_id,
        StreamEvent::Done(d) => d.request_id,
        StreamEvent::Error(e) => panic!("unexpected StreamEvent::Error: {e:?}"),
    }
}

#[test]
fn delta_frames_tile_the_requested_sample_range_with_no_gaps_or_overlaps() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();
    let request = RenderRequest {
        request_id: 11,
        scene: heavier_scene(),
        first_sample: 100,
        samples: 64,
        stream: live_progressive(0), // due on every emitter tick
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 2, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();
    let events = read_stream_until_done(&mut out_cursor);

    let mut ranges: Vec<(u32, u32)> = events
        .iter()
        .filter_map(|(e, _)| match e {
            StreamEvent::Frame(h) => Some((h.first_sample, h.samples)),
            _ => None,
        })
        .collect();
    assert!(!ranges.is_empty(), "expected at least one FRAME delta");
    ranges.sort_by_key(|r| r.0);

    let mut cursor = 100u32;
    for (first, samples) in &ranges {
        assert_eq!(
            *first, cursor,
            "gap or overlap: expected next delta to start at {cursor}, got {first}"
        );
        cursor += samples;
    }
    assert_eq!(
        cursor,
        100 + 64,
        "deltas must tile the whole requested range exactly"
    );

    assert!(matches!(
        events.last().unwrap().0,
        StreamEvent::Done(Done {
            cancelled: false,
            ..
        })
    ));
}

#[test]
fn final_only_transfer_mode_still_emits_progress() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();
    let request = RenderRequest {
        request_id: 12,
        scene: heavier_scene(),
        first_sample: 0,
        samples: 64,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 2, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();
    let events = read_stream_until_done(&mut out_cursor);

    let frame_count = events
        .iter()
        .filter(|(e, _)| matches!(e, StreamEvent::Frame(_)))
        .count();
    assert_eq!(
        frame_count, 1,
        "FinalOnly must send exactly one FRAME, covering the whole request"
    );

    let progress_count = events
        .iter()
        .filter(|(e, _)| matches!(e, StreamEvent::Progress(_)))
        .count();
    assert!(
        progress_count >= 1,
        "FinalOnly must still emit PROGRESS on the cadence even though FRAME only arrives once"
    );

    assert!(matches!(
        events.last().unwrap().0,
        StreamEvent::Done(Done {
            cancelled: false,
            ..
        })
    ));
}

#[test]
fn cancel_mid_stream_produces_done_cancelled_and_no_further_payload() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();
    let request = RenderRequest {
        request_id: 13,
        scene: heavier_scene(),
        first_sample: 0,
        samples: 64,
        stream: live_progressive(0),
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::Cancel(Cancel { request_id: 13 }),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 2, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();
    let events = read_stream_until_done(&mut out_cursor);

    for (event, _) in &events {
        assert_eq!(event_request_id(event), 13);
    }
    assert!(matches!(
        events.last().unwrap().0,
        StreamEvent::Done(Done {
            cancelled: true,
            ..
        })
    ));

    // No further payload: `duplex.out` has nothing beyond what `read_stream_until_done`
    // just consumed -- `handle_connection`'s outer loop went straight back to
    // (blocking-)reading the next RenderRequest and hit EOF, writing nothing more.
    assert_eq!(
        out_cursor.position(),
        out_cursor.get_ref().len() as u64,
        "no bytes may follow DONE{{cancelled: true, ..}}"
    );
}

#[test]
fn stale_request_id_frames_are_identifiable() {
    // Two separate connections, testing something distinct from
    // `pipelined_render_request_on_one_connection_is_queued_after_an_implicit_cancel`
    // below: THIS test is about a client that has moved on to a new request_id (via
    // any means -- a fresh connection is the simplest way to construct that state in
    // a test) being able to mechanically recognize a reply carrying the OLD one as
    // stale. The pipelined-on-one-connection scenario is the realistic trigger for
    // that same mechanism and gets its own test now that it's supported rather than
    // rejected -- see this module's and `stream_emit::run_stream`'s doc comments.
    fn run_one_request(request_id: u32) -> Vec<(StreamEvent, Option<Vec<u8>>)> {
        let mut input = Vec::new();
        gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();
        let request = RenderRequest {
            request_id,
            scene: tiny_scene(),
            first_sample: 0,
            samples: 4,
            stream: final_only(0),
        };
        gemray_net::messages::write_message(
            &mut input,
            &ClientMessage::RenderRequest(Box::new(request)),
        )
        .unwrap();

        let mut duplex = DuplexHalf::new(input);
        handle_connection(&mut duplex, 1, &test_db()).unwrap();

        let mut out_cursor = Cursor::new(duplex.out);
        let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();
        read_stream_until_done(&mut out_cursor)
    }

    let first_events = run_one_request(21);
    let second_events = run_one_request(22);

    // A client tracking "current epoch = 22" (having just issued the second
    // request) can mechanically identify every event belonging to the first, now
    // STALE request_id as one to drop -- see gemray_net::messages' own docs on
    // this being exactly what request_id echoing is for.
    assert_ne!(first_events.len(), 0);
    for (event, _) in &first_events {
        assert_eq!(event_request_id(event), 21);
        assert_ne!(event_request_id(event), 22);
    }
    assert_ne!(second_events.len(), 0);
    for (event, _) in &second_events {
        assert_eq!(event_request_id(event), 22);
    }
}

/// The scenario `stream_emit::run_stream`'s pipelining support actually exists for:
/// a client writes its NEXT `RenderRequest` right behind the first, on the SAME
/// connection, without ever reading `DONE` for the first one. Without that support,
/// this is exactly the failure mode `poll_for_client_message`'s own doc comment warns
/// about (bytes that don't decode as `Cancel`, connection torn down with a spurious
/// error) -- `handle_connection(..)` returning `Ok(())` here (rather than this test
/// `unwrap()`-panicking on an `Err`) is itself part of what this test proves.
#[test]
fn pipelined_render_request_on_one_connection_is_queued_after_an_implicit_cancel() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();

    // Heavy enough that the emitter gets multiple chances to poll (and so observe
    // the pipelined request) before the tracer finishes on its own -- see
    // `heavier_scene`'s own doc comment.
    let first_request = RenderRequest {
        request_id: 41,
        scene: heavier_scene(),
        first_sample: 0,
        samples: 64,
        stream: live_progressive(0),
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(first_request)),
    )
    .unwrap();

    // The pipelined request, written immediately behind the first -- no DONE read
    // in between. Deliberately a different (smaller, faster) scene so a leaked
    // buffer from the first request -- the wrong pixel count -- would show up as a
    // payload-size mismatch rather than silently passing.
    let second_request = RenderRequest {
        request_id: 42,
        scene: tiny_scene(),
        first_sample: 0,
        samples: 4,
        stream: final_only(0),
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(second_request)),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 2, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();

    // The first request's reply: implicitly cancelled, exactly like an explicit
    // CANCEL -- every event (if any went out before the pipelined request was
    // noticed) carries request_id 41, and it ends in DONE { cancelled: true }.
    let first_events = read_stream_until_done(&mut out_cursor);
    for (event, _) in &first_events {
        assert_eq!(event_request_id(event), 41);
    }
    assert!(
        matches!(
            first_events.last().unwrap().0,
            StreamEvent::Done(Done {
                cancelled: true,
                ..
            })
        ),
        "the superseded request must still get a proper DONE {{ cancelled: true }}, \
         not be silently dropped"
    );

    // The pipelined request's reply, immediately following on the same connection --
    // no round trip through DONE was needed to get here. FinalOnly, so exactly one
    // FRAME (correctly sized for ITS OWN scene, not the first request's) followed by
    // DONE { cancelled: false }.
    let second_events = read_stream_until_done(&mut out_cursor);
    let frames: Vec<_> = second_events
        .iter()
        .filter_map(|(e, p)| match e {
            StreamEvent::Frame(h) => Some((h, p.as_ref().unwrap())),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 1);
    let (header, payload) = frames[0];
    assert_eq!(header.request_id, 42);
    assert_eq!(header.samples, 4);
    // 4x4 (tiny_scene), not 24x24 (heavier_scene) -- proves the second request's
    // accumulation started completely fresh rather than inheriting anything from
    // the cancelled first request's (differently-sized) buffers.
    assert_eq!(payload.len(), 4 * 4 * gemray_net::radiance::BYTES_PER_PIXEL);
    for (event, _) in &second_events {
        assert_eq!(event_request_id(event), 42);
    }
    assert!(matches!(
        second_events.last().unwrap().0,
        StreamEvent::Done(Done {
            cancelled: false,
            ..
        })
    ));

    // The connection stayed open and well-formed end to end: nothing beyond the
    // two requests' worth of events was written, and handle_connection returned
    // Ok(()) above rather than aborting on a decode error.
    assert_eq!(
        out_cursor.position(),
        out_cursor.get_ref().len() as u64,
        "no bytes may follow the second request's DONE"
    );
}

#[test]
fn preview_frames_never_enter_the_full_resolution_accumulator_path() {
    let mut input = Vec::new();
    gemray_net::messages::write_message(&mut input, &handshake::local_hello()).unwrap();

    let scene = tiny_scene(); // 4x4
    let mut stream_cfg = live_progressive(0);
    stream_cfg.preview = Some(gemray_net::messages::PreviewConfig {
        width: 2,
        height: 2,
    });
    let request = RenderRequest {
        request_id: 31,
        scene: scene.clone(),
        first_sample: 0,
        samples: 8,
        stream: stream_cfg,
    };
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    handle_connection(&mut duplex, 1, &test_db()).unwrap();

    let mut out_cursor = Cursor::new(duplex.out);
    let _welcome: Welcome = gemray_net::messages::read_message(&mut out_cursor).unwrap();
    let events = read_stream_until_done(&mut out_cursor);

    let previews: Vec<_> = events
        .iter()
        .filter_map(|(e, p)| match e {
            StreamEvent::Preview(h) => Some((h, p.as_ref().unwrap())),
            _ => None,
        })
        .collect();
    assert!(
        !previews.is_empty(),
        "expected at least one PREVIEW with a preview configured"
    );

    for (header, payload) in &previews {
        assert_eq!(header.width, 2);
        assert_eq!(header.height, 2);

        // Structurally distinct from a full-resolution FRAME payload: decoding it
        // against the SCENE's own full resolution must fail with a length
        // mismatch -- proving a PREVIEW payload can never be silently fed into the
        // full-resolution accumulator path a FRAME goes through.
        let full_res_attempt = gemray_net::radiance::decode(payload, scene.width, scene.height);
        assert!(
            matches!(
                full_res_attempt,
                Err(gemray_net::radiance::RadianceError::LengthMismatch { .. })
            ),
            "{full_res_attempt:?}"
        );

        // It DOES decode correctly at its own declared (reduced) resolution -- it's
        // valid radiance data, just never for the full-resolution accumulator.
        assert!(gemray_net::radiance::decode(payload, header.width, header.height).is_ok());
    }
}

// ---- Mutual-TLS tests -------------------------------------------------------
//
// Each test builds a real (if temporary, throwaway) private CA and issues real
// certificates via `crate::pki`, then drives a real TLS handshake over a real
// loopback `TcpStream` -- nothing here is mocked at the `rustls` layer, since the
// whole point is to catch exactly the mistakes a mock would paper over (a missing
// SAN, a wrong trust anchor, an allowlist that isn't actually consulted).
mod mtls;
