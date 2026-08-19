import 'package:flutter/material.dart';

import '../application/classification_controller.dart';
import '../domain/backend_descriptor.dart';
import 'widgets/benchmark_card.dart';
import 'widgets/prediction_card.dart';
import 'widgets/runtime_card.dart';
import 'widgets/section_card.dart';
import 'widgets/timings_card.dart';

/// The single screen of the PoC.
///
/// Deliberately shows values, not explanations: image, prediction, latency,
/// model and runtime facts. The reasoning lives in `docs/`.
///
/// It talks only to [ClassificationController] and to domain types; there is no
/// import of `flutter_litert` anywhere in `lib/ui/`.
class ClassificationPage extends StatelessWidget {
  const ClassificationPage({required this.controller, super.key});

  final ClassificationController controller;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, _) {
        final result = controller.result;
        final report = controller.runtimeReport;
        final benchmark = controller.benchmark;
        final error = controller.errorMessage;

        return Scaffold(
          appBar: AppBar(
            title: const Text('On-Device Image Classification'),
            bottom: controller.isBusy
                ? const PreferredSize(
                    preferredSize: Size.fromHeight(3),
                    child: LinearProgressIndicator(minHeight: 3),
                  )
                : null,
          ),
          body: ListView(
            padding: const EdgeInsets.only(bottom: 28),
            children: [
              _ImageSection(controller: controller),
              _BackendSelector(controller: controller),
              _Actions(controller: controller),
              if (error != null)
                _ErrorCard(message: error, detail: controller.errorDetail),
              if (result != null) PredictionCard(result: result),
              if (result != null)
                TimingsCard(
                  timings: result.timings,
                  initialization: controller.initializationTime,
                ),
              if (benchmark != null) BenchmarkCard(report: benchmark),
              if (report != null)
                RuntimeCard(report: report, spec: controller.spec),
            ],
          ),
        );
      },
    );
  }
}

class _BackendSelector extends StatelessWidget {
  const _BackendSelector({required this.controller});

  final ClassificationController controller;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: DropdownButtonFormField<BackendDescriptor>(
        initialValue: controller.backend,
        isExpanded: true,
        decoration: const InputDecoration(
          labelText: 'Backend',
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        ),
        items: [
          for (final b in controller.backends)
            DropdownMenuItem(
              value: b,
              child: Text(b.label, overflow: TextOverflow.ellipsis),
            ),
        ],
        onChanged: controller.isBusy
            ? null
            : (value) {
                if (value != null) controller.selectBackend(value);
              },
      ),
    );
  }
}

class _ImageSection extends StatelessWidget {
  const _ImageSection({required this.controller});

  final ClassificationController controller;

  @override
  Widget build(BuildContext context) {
    final bytes = controller.imageBytes;
    return SectionCard(
      title: 'Input',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          AspectRatio(
            aspectRatio: 4 / 3,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(6),
              ),
              child: bytes == null
                  ? const Center(child: Text('No image selected'))
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: Image.memory(bytes, fit: BoxFit.contain),
                    ),
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final sample in ClassificationController.samples)
                ChoiceChip(
                  label: Text(sample.label),
                  selected: controller.image?.source == sample.assetPath,
                  onSelected: controller.isBusy
                      ? null
                      : (_) => controller.loadSample(sample),
                ),
              ActionChip(
                avatar: const Icon(Icons.photo_library_outlined, size: 16),
                label: const Text('Gallery'),
                onPressed: controller.isBusy ? null : controller.pickImage,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Actions extends StatelessWidget {
  const _Actions({required this.controller});

  final ClassificationController controller;

  @override
  Widget build(BuildContext context) {
    final enabled = controller.canRun;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Row(
        children: [
          Expanded(
            child: FilledButton.icon(
              onPressed: enabled ? controller.classify : null,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Run inference'),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: enabled ? () => controller.runBenchmark() : null,
              icon: const Icon(Icons.speed),
              label: const Text('Benchmark 30×'),
            ),
          ),
        ],
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message, this.detail});

  final String message;
  final String? detail;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SectionCard(
      title: 'Error',
      tone: theme.colorScheme.errorContainer,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            message,
            style: theme.textTheme.bodyLarge
                ?.copyWith(color: theme.colorScheme.onErrorContainer),
          ),
          if (detail != null) ...[
            const SizedBox(height: 6),
            Text(
              detail!,
              style: theme.textTheme.bodySmall?.copyWith(
                fontFamily: 'monospace',
                color: theme.colorScheme.onErrorContainer,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
