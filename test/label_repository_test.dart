import 'package:flutter_test/flutter_test.dart';
import 'package:on_device_ai/data/asset_sources.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/domain/ml_exceptions.dart';

import 'support/fixture.dart';

void main() {
  final spec = ModelCatalog.mobileNetV2Float32;

  LabelRepository repoReturning(String text) =>
      LabelRepository(loader: (_) async => text);

  test('loads the real shipped label file and validates its length', () async {
    final onDisk = readLabels();
    final labels =
        await LabelRepository(loader: (_) async => onDisk.join('\n')).load(spec);
    expect(labels, hasLength(1001));
    expect(labels.first, 'background');
    expect(labels[1], 'tench');
    expect(labels.last, 'toilet tissue');
  });

  test('tolerates a trailing newline', () async {
    final text = '${readLabels().join('\n')}\n';
    final labels = await repoReturning(text).load(spec);
    expect(labels, hasLength(1001));
  });

  test('rejects a label file with the wrong number of entries', () async {
    final text = List.generate(1000, (i) => 'c$i').join('\n');
    expect(
      () => repoReturning(text).load(spec),
      throwsA(isA<LabelSetException>()),
    );
  });

  test('rejects a blank line in the middle, which would shift classes',
      () async {
    final lines = readLabels();
    lines[500] = '   ';
    expect(
      () => repoReturning(lines.join('\n')).load(spec),
      throwsA(
        isA<LabelSetException>()
            .having((e) => e.message, 'message', contains('index 500')),
      ),
    );
  });

  test('rejects an empty label file', () async {
    expect(
      () => repoReturning('').load(spec),
      throwsA(isA<LabelSetException>()),
    );
  });

  test('wraps a loader failure as a LabelSetException', () async {
    final repo = LabelRepository(loader: (_) async => throw StateError('boom'));
    expect(() => repo.load(spec), throwsA(isA<LabelSetException>()));
  });

  test('caches by asset path so switching backends reloads nothing', () async {
    var calls = 0;
    final repo = LabelRepository(loader: (_) async {
      calls++;
      return readLabels().join('\n');
    });
    await repo.load(spec);
    await repo.load(spec);
    // Both bundled models share one label asset, so this is one read total.
    await repo.load(ModelCatalog.mobileNetV1Uint8);
    expect(calls, 1);
  });
}
