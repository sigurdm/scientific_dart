Choose one of the following tasks to work on (by random choice, use awk as a random dice rolling tool `awk -v min=1 -v max=5 'BEGIN{srand(); print int(min+rand()*(max-min+1))}'`).r
After being done reschedule to do another task in 1 minute with the following prompt:

```
Look in TASKS.md and follow the instructions there to find a task to work on.
```

1. Improve coverage. Measure with `dart tool/generate_coverage.dart` script. Find code missing coverage and add test cases. If you uncover problems, try to fix them and write a test case for the fix. Add description of the fix to DONE.md. If the tests are really hard to get green, just add the test cases and leave a note in FINDINGS.md about the problem.
Write a note in DONE.md about what was done, how hard it was, and if there were any notable problems, what the coverage was before and after the fix.
2. Review code
  find some files in the repo and go through them with a keen eye to see if there are any issues. Report findings in FINDINGS.md.
3. Fix one or more of the issues you found from FINDINGS.md. Remove the problem from findings, and leave a description of the work in DONE.md. If there are questions you cannot solve. Describe it in QUESTIONS.md for me to help you async.
If any of the questions have been answered, then remove the questions from QUESTIONS.md and try to continue that thread of investigation. 
4. Try to look holistically at num_dart and think about what could be improved. Write down your ideas in FINDINGS.md. Compare with what numpy can do and see what's missing.
5. Optimization: Find a slow benchmark (relative to numpy) and try to catch up. Or if you have any other ideas how to improve the performance, try it out. First write a benchmark to show the performance difference, fix it, then write a benchmark to show the improvement. Consider using a profiler to find bottlenecks if they are not obvious. 