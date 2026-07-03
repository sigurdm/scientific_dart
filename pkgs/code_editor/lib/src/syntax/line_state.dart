import 'package:meta/meta.dart';

/// Contract representing the lexer's state at the end of a line.
@immutable
abstract class LineState {
  const LineState();

  /// Returns true if this state represents the default/initial state.
  bool get isInitial;
}

/// Default initial line state.
@immutable
class EmptyLineState extends LineState {
  const EmptyLineState();

  @override
  bool get isInitial => true;

  @override
  bool operator ==(Object other) => other is EmptyLineState;

  @override
  int get hashCode => 0;

  @override
  String toString() => 'EmptyLineState';
}

/// Stack-based line state representing active rules / scopes pushed during tokenization.
@immutable
class StackLineState extends LineState {
  final List<String> stack;

  const StackLineState(this.stack);

  @override
  bool get isInitial => stack.isEmpty;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! StackLineState) return false;
    if (stack.length != other.stack.length) return false;
    for (int i = 0; i < stack.length; i++) {
      if (stack[i] != other.stack[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hashAll(stack);

  @override
  String toString() => 'StackLineState($stack)';
}
