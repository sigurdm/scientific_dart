import 'dart:ffi';
import 'dart:typed_data';

void main() {
  final list = Float64List(10);
  // Checking if .address exists via LSP or compilation check
  final p = list.address;
}
