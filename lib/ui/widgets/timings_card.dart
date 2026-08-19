import 'package:flutter/material.dart';

import '../../domain/prediction.dart';
import 'section_card.dart';

/// Per-stage cost of the last prediction.
class TimingsCard extends StatelessWidget {
  const TimingsCard({
    required this.timings,
    required this.initialization,
    super.key,
  });

  final PhaseTimings timings;
  final Duration? initialization;

  static String ms(Duration d) =>
      '${(d.inMicroseconds / 1000).toStringAsFixed(1)} ms';

  @override
  Widget build(BuildContext context) {
    final native = timings.nativeInference;
    return SectionCard(
      title: 'Latency (last run)',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (initialization != null)
            FactRow('Model init', ms(initialization!)),
          FactRow('Preprocess', ms(timings.preprocess)),
          FactRow('Inference', ms(timings.inference)),
          if (native != null)
            FactRow('  ↳ native invoke', ms(native)),
          FactRow('Postprocess', ms(timings.postprocess)),
          const Divider(height: 16),
          FactRow('Total', ms(timings.total)),
          const SizedBox(height: 6),
          Text(
            native == null
                ? 'Inference is wall-clock around the runtime call, so it '
                    'includes the FFI hop and the isolate round trip.'
                : 'The native invoke is what the runtime itself reports; the '
                    'difference is FFI and buffer-copy overhead.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
          ),
        ],
      ),
    );
  }
}
