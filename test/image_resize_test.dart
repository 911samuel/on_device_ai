import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:on_device_ai/data/image_preprocessor.dart';

import 'support/fixture.dart';

/// Locks in the resampling decision that `ImagePreprocessor` documents.
///
/// The reference tensors in the fixture come from Pillow's BILINEAR resize,
/// which antialiases when downscaling. Dart's `Interpolation.linear` does not,
/// so on a 2–3× downscale it produced visibly different pixels — enough to
/// reorder the low-confidence tail of the top-5. This test measures the
/// agreement per image and fails if the filter choice regresses.
void main() {
  final fixture = ReferenceFixture.load();
  final probe = fixture.json['resize_probe'] as Map<String, dynamic>;
  final images = probe['images'] as Map<String, dynamic>;

  /// Mean absolute difference, in 8-bit levels, between Dart's resized bytes and
  /// the reference's at the fixture's sampled positions.
  ({double mae, int worst}) agreement(String name, img.Interpolation interp) {
    final data = images[name] as Map<String, dynamic>;
    final indices = (data['sample_indices'] as List).cast<int>();
    final expected = (data['sample_values'] as List).cast<int>();
    final decoded = img.decodeImage(readRepoFile('assets/images/$name'))!;
    final resized = img.copyResize(decoded,
        width: 224, height: 224, interpolation: interp);
    final bytes = resized.getBytes(order: img.ChannelOrder.rgb);
    var sum = 0.0;
    var worst = 0;
    for (var i = 0; i < indices.length; i++) {
      final diff = (bytes[indices[i]] - expected[i]).abs();
      sum += diff;
      if (diff > worst) worst = diff;
    }
    return (mae: sum / indices.length, worst: worst);
  }

  img.Interpolation chosenFor(String name) {
    final decoded = img.decodeImage(readRepoFile('assets/images/$name'))!;
    return ImagePreprocessor.selectInterpolation(
      sourceWidth: decoded.width,
      sourceHeight: decoded.height,
      targetWidth: 224,
      targetHeight: 224,
    );
  }

  test('a genuine downscale selects area-averaging', () {
    // 512x600 -> 224x224 is a 2.3x shrink; 700x577 is 3.1x.
    expect(chosenFor('grace_hopper.jpg'), img.Interpolation.average);
    expect(chosenFor('labrador.jpg'), img.Interpolation.average);
  });

  test('a near-1:1 resize selects bilinear', () {
    // 320x213 -> 224x224: 1.43x on x, and an upscale on y.
    expect(chosenFor('cat_on_snow.jpg'), img.Interpolation.linear);
  });

  test('the selected filter beats the alternatives on every sample image', () {
    for (final name in images.keys) {
      final chosen = chosenFor(name);
      final chosenMae = agreement(name, chosen).mae;
      for (final other in img.Interpolation.values) {
        if (other == chosen) continue;
        expect(
          chosenMae,
          lessThanOrEqualTo(agreement(name, other).mae),
          reason: 'for $name, ${other.name} agrees with the reference better '
              'than the selected ${chosen.name}; the threshold in '
              'ImagePreprocessor.selectInterpolation needs revisiting',
        );
      }
    }
  });

  test('agreement with the reference stays within measured bounds', () {
    // Recorded from a real run; these are regression guards, not aspirations.
    const budgets = {
      'grace_hopper.jpg': 4.5,
      'labrador.jpg': 3.7,
      'cat_on_snow.jpg': 2.5,
    };
    budgets.forEach((name, budget) {
      final result = agreement(name, chosenFor(name));
      expect(result.mae, lessThan(budget),
          reason: '$name resampling agreement regressed to ${result.mae}');
      // A single-pixel outlier of a few dozen levels is expected at edges;
      // hundreds would mean a layout or channel-order error.
      expect(result.worst, lessThan(60));
    });
  });

  test('the preprocessor reports which filter it used', () {
    const preprocessor = ImagePreprocessor();
    final prepared = preprocessor.prepare(
      inputImageFor('grace_hopper.jpg'),
      specForFloatModel(),
    );
    expect(prepared.interpolation, 'average');

    final exact = preprocessor.prepare(
      inputImageFor('calibration_224.png'),
      specForFloatModel(),
    );
    expect(exact.interpolation, 'none',
        reason: 'an image already at the model input size must not be resampled');
  });
}
