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
      title: 'Latency',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FactRow('Preprocess', ms(timings.preprocess)),
          FactRow('Inference', ms(timings.inference)),
          if (native != null) FactRow('  native invoke', ms(native)),
          FactRow('Postprocess', ms(timings.postprocess)),
          const Divider(height: 16),
          FactRow('Total', ms(timings.total)),
          if (initialization != null)
            FactRow('Model init', ms(initialization!)),
        ],
      ),
    );
  }
}
