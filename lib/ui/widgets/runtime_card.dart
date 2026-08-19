import 'package:flutter/material.dart';

import '../../domain/model_spec.dart';
import '../../domain/prediction.dart';
import 'section_card.dart';

/// Where the honesty lives: what was asked of the hardware, what the runtime
/// says it actually got, and whether that claim was verified.
class RuntimeCard extends StatelessWidget {
  const RuntimeCard({required this.report, required this.spec, super.key});

  final RuntimeReport report;
  final ModelSpec spec;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final acc = report.accelerators;

    // Wording matters here. A verified deviation from the plain-CPU reference
    // proves a *different compute path* ran; it does not identify the silicon.
    final (String verdict, Color color) = switch (acc.accelerationProven) {
      true => ('Distinct compute path verified (silicon not identified)',
          Colors.green.shade700),
      false => ('Silent CPU fallback detected', Colors.orange.shade800),
      null => ('Acceleration not verified', theme.colorScheme.onSurfaceVariant),
    };

    return SectionCard(
      title: 'Runtime & hardware',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FactRow('Runtime', report.runtimeName),
          FactRow('Native version', report.runtimeVersion, mono: true),
          FactRow('API', report.apiName),
          FactRow('Execution', report.executionMode),
          if (acc.delegateName != null)
            FactRow('Delegate', acc.delegateName!),
          const Divider(height: 18),
          FactRow('Requested', acc.describe(acc.requested)),
          FactRow(
            'Effective',
            '${acc.describe(acc.effective)}'
            '${acc.wasNarrowed ? '  ← narrowed by the runtime' : ''}',
          ),
          FactRow('Whole graph', acc.fullGraphAccelerated ? 'yes' : 'no'),
          const SizedBox(height: 8),
          Row(
            children: [
              Icon(
                switch (acc.accelerationProven) {
                  true => Icons.verified,
                  false => Icons.warning_amber_rounded,
                  null => Icons.help_outline,
                },
                size: 18,
                color: color,
              ),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  verdict,
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(color: color, fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          if (acc.verificationSummary != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                acc.verificationSummary!,
                style: theme.textTheme.bodySmall,
              ),
            ),
          if (acc.notes.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                acc.notes,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
          const Divider(height: 18),
          FactRow('Model', spec.displayName),
          FactRow(
            'Input tensor',
            '${spec.inputShape} ${spec.inputType.name} '
            '(${spec.inputByteCount} B)',
            mono: true,
          ),
          FactRow(
            'Output tensor',
            '${spec.outputShape} ${spec.outputType.name} '
            '(${spec.outputByteCount} B)',
            mono: true,
          ),
          FactRow('Normalisation', _normalisation(spec)),
          FactRow('Asset', spec.modelAsset, mono: true),
        ],
      ),
    );
  }

  static String _normalisation(ModelSpec spec) => switch (spec.normalization) {
        InputNormalization.minusOneToOne => 'byte / 127.5 − 1 → [−1, 1]',
        InputNormalization.zeroToOne => 'byte / 255 → [0, 1]',
        InputNormalization.rawUint8 =>
          'raw 0..255; the graph applies scale ${spec.inputQuantization.scale} '
              'zero-point ${spec.inputQuantization.zeroPoint}',
      };
}
