import 'dart:io';

/// Outcome of asking the OS for an outbound TCP socket.
class NetworkProbeResult {
  const NetworkProbeResult({
    required this.socketOpened,
    required this.detail,
    required this.elapsed,
  });

  /// True when the app successfully opened a connection.
  final bool socketOpened;

  /// What happened, in plain language, including the error class when it failed.
  final String detail;

  final Duration elapsed;
}

/// Asks the platform for a real outbound socket, so the app's network capability
/// is demonstrated rather than asserted.
///
/// This exists because "it's offline because LiteRT is on-device" is not
/// evidence. The strong result is on an Android **release** build: this project
/// does not declare `android.permission.INTERNET`, so the kernel refuses the
/// socket and this probe fails — while inference keeps working. That is a
/// capability proof, not a promise.
///
/// Two honest caveats, both surfaced in the UI:
///  * In debug/profile builds Flutter's tooling manifest adds INTERNET for the
///    Dart VM service, so the probe will succeed there. Only release is
///    meaningful.
///  * iOS has no equivalent install-time network permission, so a successful
///    probe on iOS says nothing about whether inference used the network. That
///    is established separately by observing that the process holds no sockets
///    during inference (see `docs/OFFLINE_VERIFICATION.md`).
class NetworkSelfTest {
  const NetworkSelfTest({
    this.host = '1.1.1.1',
    this.port = 443,
    this.timeout = const Duration(seconds: 3),
  });

  final String host;
  final int port;
  final Duration timeout;

  Future<NetworkProbeResult> probe() async {
    final watch = Stopwatch()..start();
    Socket? socket;
    try {
      socket = await Socket.connect(host, port, timeout: timeout);
      watch.stop();
      return NetworkProbeResult(
        socketOpened: true,
        detail: 'Opened a TCP connection to $host:$port. This build is allowed '
            'to use the network — so "offline" must be shown another way '
            '(see docs/OFFLINE_VERIFICATION.md).',
        elapsed: watch.elapsed,
      );
    } on Object catch (error) {
      watch.stop();
      return NetworkProbeResult(
        socketOpened: false,
        detail: 'The OS refused an outbound socket to $host:$port '
            '(${error.runtimeType}). Inference below still runs, which means it '
            'cannot be reaching a server.',
        elapsed: watch.elapsed,
      );
    } finally {
      try {
        socket?.destroy();
      } on Object {
        /* ignore */
      }
    }
  }
}
