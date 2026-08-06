import 'dart:io';
import 'package:puppeteer/puppeteer.dart';

void main() async {
  print('Launching browser...');
  final browser = await puppeteer.launch(
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  );
  final page = await browser.newPage();

  page.onConsole.listen((msg) {
    print('CONSOLE [${msg.type}]: ${msg.text}');
  });

  page.onError.listen((err) {
    print('PAGE ERROR: $err');
  });

  print('Navigating to http://localhost:8080 ...');
  try {
    await page.goto('http://localhost:8080', wait: Until.networkIdle);
  } catch (e) {
    print('Navigation error: $e');
  }

  await Future.delayed(Duration(seconds: 3));

  final screenshot = await page.screenshot();
  File('/tmp/notebook_debug.png').writeAsBytesSync(screenshot);
  print('Saved debug screenshot to /tmp/notebook_debug.png');

  final statusText = await page.evaluate(
    "() => document.getElementById('statusText') ? document.getElementById('statusText').innerText : 'NO STATUS'",
  );
  print('Status Text on Page: $statusText');

  final cellCount = await page.evaluate(
    "() => document.querySelectorAll('.cell').length",
  );
  print('Cell Count on Page: $cellCount');

  await browser.close();
}
