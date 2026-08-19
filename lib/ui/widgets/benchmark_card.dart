import 'package:flutter/material.dart';

import '../../application/benchmark.dart';
import 'section_card.dart';

/// Cold vs warm distribution over N runs. Numbers only.
class BenchmarkCard extends StatelessWidget {
  const BenchmarkCard({required this.report, super.key});

  final BenchmarkReport report;

  static String _ms(Duration d) =>
      '${(d.inMicroseconds / 1000).toStringAsFixed(1)} ms';

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: 'Benchmark · ${report.iterations} runs',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FactRow('Cold run', _ms(report.cold.total)),
          FactRow('Warm median', _ms(report.warmTotal.median)),
          const Divider(height: 16),
          FactRow('Inference', report.warmInference.describeMs(), mono: true),
          FactRow('Total', report.warmTotal.describeMs(), mono: true),
        ],
      ),
    );
  }
}
