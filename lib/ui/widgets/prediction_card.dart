import 'package:flutter/material.dart';

import '../../domain/prediction.dart';
import 'section_card.dart';

/// Top-1 prominently, then the rest of the top-K with confidence bars.
class PredictionCard extends StatelessWidget {
  const PredictionCard({required this.result, super.key});

  final PredictionResult result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final top = result.top;
    return SectionCard(
      title: 'Prediction',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            top.label,
            style: theme.textTheme.headlineSmall
                ?.copyWith(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 2),
          Text(
            '${top.confidencePercent} confidence · class #${top.classIndex}',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          if (result.verdict != ConfidenceVerdict.decisive)
            _ConfidenceNote(verdict: result.verdict),
          const Divider(height: 22),
          for (final p in result.predictions.skip(1))
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                children: [
                  Expanded(
                    flex: 5,
                    child: Text(p.label, style: theme.textTheme.bodySmall),
                  ),
                  Expanded(
                    flex: 4,
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: LinearProgressIndicator(
                        value: p.confidence.clamp(0.0, 1.0),
                        minHeight: 6,
                        backgroundColor:
                            theme.colorScheme.surfaceContainerHighest,
                      ),
                    ),
                  ),
                  SizedBox(
                    width: 56,
                    child: Text(
                      p.confidencePercent,
                      textAlign: TextAlign.right,
                      style: theme.textTheme.bodySmall,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// Shown when the top-1 label does not clear the 50% mark the test suite uses
/// for a decisive prediction.
///
/// It exists because these models cannot abstain: they always return one of
/// 1,001 ImageNet classes, and that list contains no person, building, road or
/// food class. A photo of something outside the list still produces a
/// confident-looking label, so the honest thing is to say the score is weak
/// rather than let the headline stand unqualified.
class _ConfidenceNote extends StatelessWidget {
  const _ConfidenceNote({required this.verdict});

  final ConfidenceVerdict verdict;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final inconclusive = verdict == ConfidenceVerdict.inconclusive;
    final tint = inconclusive
        ? theme.colorScheme.errorContainer
        : theme.colorScheme.surfaceContainerHighest;
    final ink = inconclusive
        ? theme.colorScheme.onErrorContainer
        : theme.colorScheme.onSurfaceVariant;

    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: tint,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            inconclusive ? Icons.help_outline : Icons.info_outline,
            size: 16,
            color: ink,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  verdict.headline,
                  style: theme.textTheme.labelLarge?.copyWith(color: ink),
                ),
                const SizedBox(height: 2),
                Text(
                  verdict.explanation,
                  style: theme.textTheme.bodySmall?.copyWith(color: ink),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
