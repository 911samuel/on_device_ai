import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'application/classification_controller.dart';
import 'domain/input_image.dart';
import 'ui/classification_page.dart';

void main() {
  runApp(const OnDeviceAiApp());
}

/// Reads a gallery image into raw bytes.
///
/// The picker is wired in here, at the app's edge, and handed to the controller
/// as a function. That keeps `image_picker` out of the ML and application layers
/// and makes the controller testable without a platform channel.
Future<InputImage?> pickImageFromGallery() async {
  final picked = await ImagePicker().pickImage(source: ImageSource.gallery);
  if (picked == null) return null;
  final bytes = await picked.readAsBytes();
  return InputImage(bytes: bytes, source: 'gallery:${picked.name}');
}

class OnDeviceAiApp extends StatefulWidget {
  const OnDeviceAiApp({super.key});

  @override
  State<OnDeviceAiApp> createState() => _OnDeviceAiAppState();
}

class _OnDeviceAiAppState extends State<OnDeviceAiApp> {
  late final ClassificationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = ClassificationController(imagePicker: pickImageFromGallery);
    // Loading and compiling a 13 MB model is not instant; start it after the
    // first frame so the UI is on screen and can show progress.
    WidgetsBinding.instance.addPostFrameCallback((_) => _controller.start());
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'On-Device AI · LiteRT',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF00639B)),
        useMaterial3: true,
        visualDensity: VisualDensity.compact,
      ),
      home: ClassificationPage(controller: _controller),
    );
  }
}
