import 'dart:convert';
import 'dart:io';

/// Orchestrates multiple agents by managing a task registry and preventing collisions.
void main(List<String> args) async {
  if (args.isEmpty) {
    printUsage();
    return;
  }

  final command = args[0];
  final registryFile = File('ACTIVE_TASKS.json');

  switch (command) {
    case 'claim':
      await handleClaim(args.sublist(1), registryFile);
      break;
    case 'release':
      await handleRelease(args.sublist(1), registryFile);
      break;
    case 'list':
      await handleList(registryFile);
      break;
    default:
      print('Unknown command: $command');
      printUsage();
  }
}

void printUsage() {
  print('Usage: dart tool/agent_orchestrator.dart <command> [args]');
  print('Commands:');
  print('  claim <agent_id> <task_id> [finding_id]  Claims a task/finding');
  print(
    '  release <agent_id>                       Releases all tasks for an agent',
  );
  print('  list                                     Lists active tasks');
}

Future<void> handleClaim(List<String> args, File registryFile) async {
  if (args.length < 2) {
    print('Usage: claim <agent_id> <task_id> [finding_id]');
    exit(1);
  }

  final agentId = args[0];
  final taskId = args[1];
  final findingId = args.length > 2 ? args[2] : null;

  final registry = await readRegistry(registryFile);

  // Check if task is already claimed
  for (final entry in registry.values) {
    if (entry['task_id'] == taskId) {
      if (findingId == null || entry['finding_id'] == findingId) {
        print(
          'Task $taskId ${findingId != null ? "finding $findingId " : ""}is already claimed by ${entry['agent_id']}',
        );
        exit(1);
      }
    }
  }

  registry[agentId] = {
    'agent_id': agentId,
    'task_id': taskId,
    'finding_id': findingId,
    'claimed_at': DateTime.now().toIso8601String(),
  };

  await writeRegistry(registryFile, registry);
  print(
    'Agent $agentId successfully claimed task $taskId${findingId != null ? " ($findingId)" : ""}',
  );
}

Future<void> handleRelease(List<String> args, File registryFile) async {
  if (args.isEmpty) {
    print('Usage: release <agent_id>');
    exit(1);
  }

  final agentId = args[0];
  final registry = await readRegistry(registryFile);

  if (registry.containsKey(agentId)) {
    final entry = registry.remove(agentId);
    await writeRegistry(registryFile, registry);
    print('Agent $agentId released task ${entry['task_id']}');
  } else {
    print('No active tasks for agent $agentId');
  }
}

Future<void> handleList(File registryFile) async {
  final registry = await readRegistry(registryFile);
  if (registry.isEmpty) {
    print('No active tasks.');
    return;
  }

  print('Active Tasks:');
  registry.forEach((agentId, data) {
    print(
      '- $agentId: Task ${data['task_id']} ${data['finding_id'] ?? ""} (Claimed at ${data['claimed_at']})',
    );
  });
}

Future<Map<String, dynamic>> readRegistry(File file) async {
  if (!await file.exists()) return {};
  final content = await file.readAsString();
  if (content.trim().isEmpty) return {};
  return jsonDecode(content) as Map<String, dynamic>;
}

Future<void> writeRegistry(File file, Map<String, dynamic> registry) async {
  await file.writeAsString(JsonEncoder.withIndent('  ').convert(registry));
}
