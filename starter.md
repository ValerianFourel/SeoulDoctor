# Start pstack in Codex Cloud

This repository includes the community Codex adaptation of pstack. The installation adds 52 project skills under `.agents/skills` and records their source versions in `skills-lock.json`.

The skills do not change the Seoul Doctor application at runtime. Codex reads them as task instructions. Some pstack workflows can run scripts or delegate work, so review a skill before you use it.

## Make the skills available

Codex discovers repository skills when a task starts. To use the installed skills in Codex Cloud:

1. Push this branch to the remote repository.
2. Start a new Codex Cloud task from the same branch.
3. Ask Codex to configure pstack:

   ```text
   Use the setup-pstack skill to configure pstack for the models
   and tools available in this Codex Cloud environment.
   ```

`setup-pstack` maps pstack roles to models that the current Codex environment supports. The Codex adaptation stores the model sheet at `~/.codex/pstack-models.md` and loads its settings through `~/.codex/AGENTS.md`.

## Enable poteto-mode

Poteto mode is opt-in for each task. Start a new task on this branch and name the skill in your request:

```text
Use the poteto-mode skill to investigate this bug, reproduce it,
implement the fix, and verify it.
```

Replace the bug description with your task. For example:

```text
Use the poteto-mode skill to review the documentation experience,
fix the problems you find, and verify the result.
```

Poteto mode then selects the relevant pstack playbook. Its workflow emphasizes a written plan, small verified changes, concise prose, and evidence from tests or the running product. It can use supporting skills such as `architect`, `how`, `swarm`, and `technical-writing` when the task needs them.

Naming `poteto-mode` in the prompt enables the workflow for that task. Installing the files does not add a `/poteto-mode` slash command to Codex Cloud, and it does not grant access to models or tools that the environment does not already provide.

## Restore or update the installation

Run these commands from the repository root.

Restore the pinned skills from `skills-lock.json`:

```bash
npx skills experimental_install
```

Update the installed project skills:

```bash
npx skills update --project --yes
```

An update can change the skill instructions and `skills-lock.json`. Review and commit both changes together.

## Verify discovery

List the skills that Codex can see:

```bash
npx skills list --agent codex
```

The output must identify the skills as project-scoped and include both `setup-pstack` and `poteto-mode`.
