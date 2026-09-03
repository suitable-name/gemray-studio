//! TEMPORARY measurement harness -- not part of the deliverable, deleted after use.
//!
//! Mirrors `apps/diagram-gui/src/bridge/remote_render.rs::connect_and_handshake` step by
//! step, timing each phase separately, against a real `gemray-worker serve` process on
//! loopback. Run with a worker already listening:
//!
//! ```text
//! cargo run -p gemray-net --release --example bench_handshake -- <addr> <ca> <cert> <key> [iterations]
//! ```

use gemray_net::client::handshake::handshake as app_handshake;
use rustls::pki_types::ServerName;
use std::{net::TcpStream, path::PathBuf, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let addr = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:18443".into());
    let ca_path = PathBuf::from(args.get(2).cloned().unwrap_or_default());
    let cert_path = PathBuf::from(args.get(3).cloned().unwrap_or_default());
    let key_path = PathBuf::from(args.get(4).cloned().unwrap_or_default());
    let iterations: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30);

    let host = addr.rsplit_once(':').map_or(addr.as_str(), |(h, _)| h);

    let mut t_load_ca = Vec::new();
    let mut t_load_certs = Vec::new();
    let mut t_load_key = Vec::new();
    let mut t_config = Vec::new();
    let mut t_connect = Vec::new();
    let mut t_tls_handshake = Vec::new();
    let mut t_app_handshake = Vec::new();
    let mut t_total = Vec::new();

    for _ in 0..iterations {
        let total_start = Instant::now();

        let t0 = Instant::now();
        let ca = gemray_net::tls::load_ca(&ca_path).expect("load_ca");
        t_load_ca.push(t0.elapsed());

        let t0 = Instant::now();
        let cert_chain = gemray_net::tls::load_certs(&cert_path).expect("load_certs");
        t_load_certs.push(t0.elapsed());

        let t0 = Instant::now();
        let key = gemray_net::tls::load_private_key(&key_path).expect("load_private_key");
        t_load_key.push(t0.elapsed());

        let t0 = Instant::now();
        let config = gemray_net::tls::client_config(ca, cert_chain, key).expect("client_config");
        t_config.push(t0.elapsed());

        let t0 = Instant::now();
        let tcp = TcpStream::connect(&addr).expect("tcp connect");
        t_connect.push(t0.elapsed());

        let server_name = ServerName::try_from(host.to_string()).expect("server name");
        let conn = rustls::ClientConnection::new(config, server_name).expect("client conn");
        let mut stream = rustls::StreamOwned::new(conn, tcp);

        let t0 = Instant::now();
        stream
            .conn
            .complete_io(&mut stream.sock)
            .expect("complete_io");
        t_tls_handshake.push(t0.elapsed());

        let t0 = Instant::now();
        let _welcome = app_handshake(&mut stream).expect("HELLO/WELCOME handshake");
        t_app_handshake.push(t0.elapsed());

        t_total.push(total_start.elapsed());
        drop(stream);
    }

    print_stats("load_ca (file read)", &t_load_ca);
    print_stats("load_certs (file read)", &t_load_certs);
    print_stats("load_private_key (file read)", &t_load_key);
    print_stats("client_config (build)", &t_config);
    print_stats("TcpStream::connect", &t_connect);
    print_stats("TLS complete_io (handshake)", &t_tls_handshake);
    print_stats("HELLO/WELCOME (app handshake)", &t_app_handshake);
    print_stats("TOTAL connect_and_handshake", &t_total);
}

fn print_stats(label: &str, samples: &[std::time::Duration]) {
    let mut micros: Vec<f64> = samples
        .iter()
        .map(std::time::Duration::as_secs_f64)
        .map(|s| s * 1e6)
        .collect();
    micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = micros.len();
    let sum: f64 = micros.iter().sum();
    let mean = sum / n as f64;
    let median = micros[n / 2];
    let min = micros[0];
    let max = micros[n - 1];
    println!(
        "{label:32} n={n:3}  min={min:9.1}us  median={median:9.1}us  mean={mean:9.1}us  max={max:9.1}us"
    );
}
