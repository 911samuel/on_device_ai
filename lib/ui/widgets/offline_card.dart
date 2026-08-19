import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../application/network_self_test.dart';
import '../../domain/model_spec.dart';
import 'section_card.dart';

/// Evidence that execution is local, rather than a badge that asserts it.
class OfflineCard extends StatelessWidget {
  const OfflineCard({
    required this.spec,
    required this.probe,
    required this.onProbe,
    required this.busy,
    super.key,
  });

  final ModelSpec spec;
  final NetworkProbeResult? probe;
  final VoidCallback onProbe;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SectionCard(
      title: 'Local execution evidence',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FactRow(
            'Weights',
            '${spec.fileSizeBytes} bytes read from the app package via '
                'rootBundle — no download path exists in this build.',
          ),
          FactRow('Build mode', kReleaseMode ? 'release' : 'debug/profile'),
          if (!kReleaseMode)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Text(
                'Note: Flutter\'s debug manifest adds android.permission.INTERNET '
                'for the Dart VM service, so the probe below only means '
                'something in a release build.',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: Colors.orange.shade800),
              ),
            ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerLeft,
            child: OutlinedButton.icon(
              onPressed: busy ? null : onProbe,
              icon: const Icon(Icons.wifi_find, size: 18),
              label: const Text('Run network self-test'),
            ),
          ),
          if (probe != null) ...[
            const SizedBox(height: 8),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  probe!.socketOpened ? Icons.public : Icons.public_off,
                  size: 18,
                  color: probe!.socketOpened
                      ? Colors.orange.shade800
                      : Colors.green.shade700,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    '${probe!.socketOpened ? 'Socket opened' : 'Socket refused'} '
                    'in ${probe!.elapsed.inMilliseconds} ms — ${probe!.detail}',
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}
