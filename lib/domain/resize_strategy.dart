/// How a non-square source image is fitted to the model's square input.
///
/// This is a genuine accuracy decision, not a formatting detail, and neither
/// option wins everywhere — which is why the app exposes both rather than
/// picking one and hoping. Measured against the Python reference on the three
/// bundled samples with MobileNetV2:
///
/// | image            | [stretch]              | [centreCrop]           |
/// |------------------|------------------------|------------------------|
/// | labrador.jpg     | Labrador retriever 0.28| Labrador retriever 0.51|
/// | cat_on_snow.jpg  | lynx 0.35              | Egyptian cat 0.45      |
/// | grace_hopper.jpg | military uniform 0.80  | mortarboard 0.52       |
///
/// The pattern: cropping helps when the subject is centred and fills the frame
/// (the Labrador nearly doubles), and hurts when the crop removes the part that
/// identifies the class (Grace Hopper's uniform is cropped away, leaving the
/// hat, so the label changes to "mortarboard").
///
/// Phone photos are usually 4:3 or 16:9, so they are distorted far more by
/// [stretch] than these near-square samples are — which is why a gallery photo
/// often scores lower than the bundled examples.
enum ResizeStrategy {
  /// Scale both axes independently to the model's input size, keeping the whole
  /// frame and distorting the aspect ratio.
  ///
  /// The default, for two reasons: it keeps the entire subject visible (nothing
  /// can be cropped out of view), and it is what `tool/reference_predict.py`
  /// does, so the committed reference fixture and the bit-level parity tests
  /// describe this path.
  stretch,

  /// The standard ImageNet evaluation recipe: scale the *short* side to
  /// `input / 0.875` (256 for a 224 model), then take the centre square.
  ///
  /// Preserves the aspect ratio at the cost of discarding the edges of the
  /// frame. This is what the published accuracy figures for these models were
  /// measured with, so it is the more faithful comparison — when the subject is
  /// where the recipe assumes it is.
  centreCrop;

  /// Fraction of the short side kept by [centreCrop]; 87.5% is the ImageNet
  /// convention (224 of 256).
  static const double centreCropRatio = 0.875;

  String get label => switch (this) {
        ResizeStrategy.stretch => 'Stretch (whole frame)',
        ResizeStrategy.centreCrop => 'Centre-crop (ImageNet recipe)',
      };
}
