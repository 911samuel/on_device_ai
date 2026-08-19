import 'package:flutter/material.dart';

import '../../application/benchmark.dart';
import 'section_card.dart';

/// Cold vs warm, with distribution rather than a single average.
class BenchmarkCard extends StatelessWidget {
  const BenchmarkCard({required this.report, super.key});

  final BenchmarkReport report;

  static String _ms(Duration d) =>
      '${(d.inMicroseconds / 1000).toStringAsFixed(1)} ms';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SectionCard(
      title: 'Benchmark · ${report.iterations} runs',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FactRow('Backend', report.backendLabel),
          FactRow('Init (once)', _ms(report.initialization)),
          const Divider(height: 16),
          FactRow('Cold run 1', _ms(report.cold.total)),
          FactRow(
            'Warm median',
            '${_ms(report.warmTotal.median)}   '
            '(cold is ${report.coldPenaltyFactor.toStringAsFixed(1)}× slower)',
          ),
          const Divider(height: 16),
          Text('Warm runs 2–${report.iterations}',
              style: theme.textTheme.bodySmall),
          const SizedBox(height: 6),
          FactRow('Preprocess', report.warmPreprocess.describeMs(), mono: true),
          FactRow('Inference', report.warmInference.describeMs(), mono: true),
          FactRow('Postprocess', report.warmPostprocess.describeMs(),
              mono: true),
          FactRow('Total', report.warmTotal.describeMs(), mono: true),
          const SizedBox(height: 8),
          Text(
            'Produced "${report.topLabel}" at '
            '${(report.topConfidence * 100).toStringAsFixed(1)}% — a fast '
            'configuration that predicts the wrong thing is not a win.',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}
