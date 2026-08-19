import 'package:flutter/material.dart';

import '../../domain/model_spec.dart';
import '../../domain/prediction.dart';
import 'section_card.dart';

/// Model and runtime facts. Values only — the reasoning behind them belongs in
/// the presentation, not on the device screen.
class RuntimeCard extends StatelessWidget {
  const RuntimeCard({required this.report, required this.spec, super.key});

  final RuntimeReport report;
  final ModelSpec spec;

  @override
  Widget build(BuildContext context) {
    final acc = report.accelerators;
    final effective = acc.describe(acc.effective);

    return SectionCard(
      title: 'Model & runtime',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FactRow('Model', spec.displayName),
          FactRow(
            'Input',
            '${spec.inputShape} ${spec.inputType.name}',
            mono: true,
          ),
          FactRow(
            'Output',
            '${spec.outputShape} ${spec.outputType.name}',
            mono: true,
          ),
          FactRow('Size', '${(spec.fileSizeBytes / 1048576).toStringAsFixed(2)} MB'),
          const Divider(height: 16),
          FactRow('Runtime', 'LiteRT ${report.runtimeVersion}', mono: true),
          FactRow('API', report.apiName),
          FactRow(
            'Accelerator',
            acc.wasNarrowed
                ? '$effective  (requested ${acc.describe(acc.requested)})'
                : effective,
          ),
          FactRow('Device', 'On-device · no network'),
        ],
      ),
    );
  }
}
