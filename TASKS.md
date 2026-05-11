# Multi-Agent Orchestration Protocol

Before starting a task, you MUST claim it and work on a dedicated branch to avoid collisions.

1. **Claim Task**: `dart tool/agent_orchestrator.dart claim <agent_id> <task_id> [finding_id]`
2. **Branching**: Create and switch to a new branch for your work:
   `git checkout -b agents/<task_id>-<agent_id>`
3. **Work**: Perform the task as described below.
4. **Log Progress**: Create a new file `done/<task_id>_<agent_id>.md`.
5. **Commit**: Commit your changes to your branch.
6. **Release Task**: `dart tool/agent_orchestrator.dart release <agent_id>`
7. **Cleanup**: Switch back to the main branch after finishing (but do not delete your branch).

---

Choose one of the following tasks to work on (by random choice, use awk as a random dice rolling tool `awk -v min=1 -v max=8 'BEGIN{srand(); print int(min+rand()*(max-min+1))}'`).
After being done with the chosen task reschedule to do another task in 10 seconds with the following prompt:

```
Check ACTIVE_TASKS.json to see what is being worked on.
Look in TASKS.md and follow the instructions there to find a task to work on.
It is important to read the file each time, as
it could be updated with new tasks or instructions.
```

1. Improve coverage. Measure with `dart pkgs/ndarray/tool/generate_coverage.dart` script. Find code missing coverage and add test cases. If you uncover problems, try to fix them and write a test case for the fix.  If the tests are really hard to get green, just add the test cases and leave a note in FINDINGS.md about the problem.
Create a note in `done/` about what was done.
2. Review code
  find some files in the repo and go through them with a keen eye to see if there are any issues. Report findings in FINDINGS.md.
3. Fix one or more of the issues you found from FINDINGS.md. 
Before starting, mark the item in FINDINGS.md as `[CLAIMED: <agent_id>]`.
After finishing, remove the problem from FINDINGS.md, and leave a description of the work in `done/`. 
If there are questions you cannot solve, describe them in QUESTIONS.md for me to help you async.
If any item in FINDINGS.md is annotated with `NEXT` chose that one. If there are multiple `NEXT` annotations, chose one randomly. If there are none, then chose any.
4. Try to look holistically at ndarray and think about what could be improved. Compare with what numpy can do and see what's missing. Write down your ideas in FINDINGS.md. 
5. Optimization: Find a slow benchmark (always write a numpy equivalent) and try to catch up. Or if you have any other ideas how to improve the performance, try it out. First write a benchmark to show the performance difference, fix it, then write a benchmark to show the improvement. Consider using a profiler to find bottlenecks if they are not obvious. Create a note in `done/` about what was done.
6. Documentation: Find a file in the repo and make sure all api members are documented and the documentation is correct and with usage examples where applicable. If not, fix it and leave a description of the work in `done/`.
7. Same as 3.
8. Same as 3.

If any of the questions have been answered, then remove the question from QUESTIONS.md and try to continue that thread of investigation.

Commit all changes after finishing a task.
Then compress context.
