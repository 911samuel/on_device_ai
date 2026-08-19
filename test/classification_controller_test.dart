import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:on_device_ai/application/classification_controller.dart';
import 'package:on_device_ai/data/backend_registry.dart';
import 'package:on_device_ai/data/model_catalog.dart';
import 'package:on_device_ai/domain/input_image.dart';
import 'package:on_device_ai/domain/ml_exceptions.dart';

import 'support/fake_on_device_model.dart';

/// The controller is exercised entirely without a native runtime, which is only
/// possible because it depends on [OnDeviceModel] rather than on LiteRT.
void main() {
  final imageBytes = Uint8List.fromList([1, 2, 3, 4]);

  ClassificationController controllerWith(
    FakeOnDeviceModel model, {
    ImagePickerFn? picker,
  }) =>
      ClassificationController(
        modelFactory: (_) => model,
        assetLoader: (_) async => imageBytes,
        imagePicker: picker,
      );

  test('start() initialises the default backend and loads a sample', () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller = controllerWith(model);

    await controller.start();

    expect(model.initializeCalls, 1);
    expect(controller.status, ControllerStatus.ready);
    expect(controller.image?.source, 'assets/images/grace_hopper.jpg');
    expect(controller.canRun, isTrue);
    expect(controller.errorMessage, isNull);
    controller.dispose();
  });

  test('a failed initialisation surfaces a user message, not a crash',
      () async {
    final model = FakeOnDeviceModel(
      spec: ModelCatalog.mobileNetV2Float32,
      initializeError: const ModelAssetMissingException('assets/models/x'),
    );
    final controller = controllerWith(model);

    await controller.start();

    expect(controller.status, ControllerStatus.error);
    expect(controller.errorMessage, contains('model file is missing'));
    expect(controller.errorDetail, contains('ModelAssetMissingException'));
    expect(controller.canRun, isFalse);
    controller.dispose();
  });

  test('an unsupported-platform failure is reported verbatim in the detail',
      () async {
    final model = FakeOnDeviceModel(
      spec: ModelCatalog.mobileNetV1Uint8,
      initializeError: const UnsupportedPlatformException(
        'CompiledModel supports float32 I/O only',
      ),
    );
    final controller = controllerWith(model);
    await controller.selectBackend(BackendRegistry.compiledV2GpuCpu);
    expect(controller.status, ControllerStatus.error);
    expect(controller.errorDetail, contains('float32 I/O only'));
    controller.dispose();
  });

  test('classify() publishes a result and returns to ready', () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller = controllerWith(model);
    await controller.start();

    var notifications = 0;
    controller.addListener(() => notifications++);
    await controller.classify();

    expect(model.predictCalls, 1);
    expect(controller.status, ControllerStatus.ready);
    expect(controller.result?.top.label, 'Labrador retriever');
    expect(controller.result?.timings.total.inMilliseconds, 15);
    expect(notifications, greaterThanOrEqualTo(2)); // running -> ready
    controller.dispose();
  });

  test('an inference failure leaves the app usable', () async {
    final model = FakeOnDeviceModel(
      spec: ModelCatalog.mobileNetV2Float32,
      predictError: const InferenceException('native invoke returned an error'),
    );
    final controller = controllerWith(model);
    await controller.start();
    await controller.classify();

    expect(controller.status, ControllerStatus.error);
    expect(controller.errorMessage, 'Inference failed on this device.');
    expect(controller.result, isNull);
    controller.dispose();
  });

  test('switching backend disposes the previous model before building the next',
      () async {
    final first = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final second = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV1Uint8);
    var built = 0;
    final controller = ClassificationController(
      modelFactory: (_) => built++ == 0 ? first : second,
      assetLoader: (_) async => imageBytes,
    );

    await controller.start();
    await controller.selectBackend(BackendRegistry.interpreterV1QuantXnnpack);

    expect(first.disposeCalls, 1);
    expect(second.initializeCalls, 1);
    expect(controller.backend.id, 'interpreter_v1q_xnnpack');
    expect(controller.spec.id, 'mobilenet_v1_uint8');
    expect(controller.status, ControllerStatus.ready);
    controller.dispose();
  });

  test('changing backend clears the previous result and benchmark', () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller = ClassificationController(
      modelFactory: (_) => model,
      assetLoader: (_) async => imageBytes,
    );
    await controller.start();
    await controller.classify();
    expect(controller.result, isNotNull);

    await controller.selectBackend(BackendRegistry.compiledV2Cpu);
    expect(controller.result, isNull);
    expect(controller.benchmark, isNull);
    controller.dispose();
  });

  test('the benchmark populates cold and warm statistics', () async {
    final model = FakeOnDeviceModel(
      spec: ModelCatalog.mobileNetV2Float32,
      inferenceDurations: const [
        Duration(milliseconds: 50),
        Duration(milliseconds: 10),
        Duration(milliseconds: 11),
      ],
    );
    final controller = controllerWith(model);
    await controller.start();

    await controller.runBenchmark(iterations: 3);

    expect(controller.benchmark, isNotNull);
    expect(controller.benchmark!.cold.inference,
        const Duration(milliseconds: 50));
    expect(controller.benchmark!.warmInference.count, 2);
    expect(controller.status, ControllerStatus.ready);
    controller.dispose();
  });

  test('a cancelled image picker leaves state untouched', () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller = controllerWith(model, picker: () async => null);
    await controller.start();
    final before = controller.image?.source;

    await controller.pickImage();

    expect(controller.image?.source, before);
    expect(controller.errorMessage, isNull);
    controller.dispose();
  });

  test('a picked image replaces the sample and clears the old result',
      () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller = controllerWith(
      model,
      picker: () async => InputImage(
        bytes: Uint8List.fromList([9, 9, 9]),
        source: 'gallery:IMG_0042.jpg',
      ),
    );
    await controller.start();
    await controller.classify();

    await controller.pickImage();

    expect(controller.image?.source, 'gallery:IMG_0042.jpg');
    expect(controller.result, isNull);
    controller.dispose();
  });

  test('a picker throwing is reported rather than propagating', () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller =
        controllerWith(model, picker: () async => throw StateError('denied'));
    await controller.start();
    await controller.pickImage();
    expect(controller.status, ControllerStatus.error);
    expect(controller.errorMessage, 'That image could not be opened.');
    controller.dispose();
  });

  test('disposing the controller releases the model\'s native memory',
      () async {
    final model = FakeOnDeviceModel(spec: ModelCatalog.mobileNetV2Float32);
    final controller = controllerWith(model);
    await controller.start();

    controller.dispose();
    await Future<void>.delayed(Duration.zero);

    expect(model.disposeCalls, 1);
  });
}
