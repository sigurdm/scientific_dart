import 'package:test/test.dart';
import 'package:notebook/notebook.dart';
import 'package:notebook/src/kernel_helper.dart';

void main() {
  test('LaTeX and Latex generate valid math-latex HTML markup', () {
    final eq1 = LaTeX(r'E = m c^2');
    expect(eq1.toHtml(), contains('class="math-latex"'));
    expect(eq1.toHtml(), contains(r'\(E = m c^2\)'));

    final eq2 = Latex(r'A x = \lambda x');
    expect(prettyFormat(eq2), contains(r'\(A x = \lambda x\)'));
  });
}
