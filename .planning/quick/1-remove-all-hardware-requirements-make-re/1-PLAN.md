---
phase: quick-1
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  # Task 1 — docs
  - README.md
  - docs/architecture/multi-gpu-strategy.md
  - docs/implementation-plan.md
  - docs/architecture/monorepo-mapping.md
  - docs/architecture/recursive-loop.md
  - docs/article/methods.md
  - docs/article/background.md
  - docs/article/discussion.md
  - libs/model-training/README.md
  - mkdocs.yml
  # Task 2 — code/config
  - services/lora-server/config.py
  - services/lora-server/config.yaml
  - services/lora-server/startup.sh
  - infra/docker-compose.yml
autonomous: true
requirements: []

must_haves:
  truths:
    - "No doc or code file prescribes specific GPU hardware (RTX 4090, Threadripper, CXL, specific VRAM amounts) as a requirement"
    - "Docs describe the system as running on any local setup with a CUDA-capable GPU, with hardware choices presented as examples or recommendations rather than requirements"
    - "lora-server config, startup script, and docker-compose are parameterized so users can configure parallelism for their own hardware"
  artifacts:
    - path: "README.md"
      provides: "Hardware-agnostic project description"
      contains: "local"
    - path: "docs/architecture/multi-gpu-strategy.md"
      provides: "GPU strategy doc reframed as configurable guidance, not hardware-specific prescription"
    - path: "services/lora-server/config.py"
      provides: "Configurable parallelism defaults without forbidden TP=2 hardcoded guard"
    - path: "services/lora-server/startup.sh"
      provides: "Startup script reading parallelism from config/env vars"
  key_links:
    - from: "services/lora-server/config.yaml"
      to: "services/lora-server/startup.sh"
      via: "startup.sh reads parallelism settings from config.yaml or env vars"
      pattern: "pipeline.parallel|PIPELINE_PARALLEL"
---

<objective>
Remove all baked-in hardware requirements from the codebase. The goal of Rune is to run on any local setup with appropriate GPU resources, not to prescribe a specific hardware configuration (2x RTX 4090, CXL, AMD Threadripper, etc.). All docs and configs should be updated to present hardware choices as examples/recommendations rather than hard requirements, and configs should be parameterized for user flexibility.

Purpose: Make the project accessible to anyone with a local CUDA-capable GPU setup, rather than requiring a specific dual-RTX-4090 configuration.
Output: Updated docs and configs across the entire repo.
</objective>

<execution_context>
@/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/workflows/execute-plan.md
@/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/templates/summary.md
</execution_context>

<context>
@README.md
@docs/architecture/multi-gpu-strategy.md
@docs/implementation-plan.md
@docs/architecture/monorepo-mapping.md
@docs/architecture/recursive-loop.md
@docs/article/methods.md
@docs/article/background.md
@docs/article/discussion.md
@libs/model-training/README.md
@mkdocs.yml
@services/lora-server/config.py
@services/lora-server/config.yaml
@services/lora-server/startup.sh
@infra/docker-compose.yml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Remove hardware prescriptions from all documentation</name>
  <files>
    README.md,
    docs/architecture/multi-gpu-strategy.md,
    docs/implementation-plan.md,
    docs/architecture/monorepo-mapping.md,
    docs/architecture/recursive-loop.md,
    docs/article/methods.md,
    docs/article/background.md,
    docs/article/discussion.md,
    libs/model-training/README.md,
    mkdocs.yml
  </files>
  <action>
    Systematically update every documentation file to remove hardware prescriptions. The guiding principle: hardware choices should be presented as EXAMPLES or RECOMMENDATIONS, not REQUIREMENTS. Specific changes per file:

    **README.md:**
    - Replace the "Hardware Requirements" section (lines 136-153) with a "Hardware" section that says Rune runs on any local machine with a CUDA-capable GPU. Mention that multi-GPU setups can use pipeline parallelism for better throughput. Remove the specific table listing RTX 4090, CXL, Threadripper, CUDA 12.8+, PyTorch 2.9+. Instead, state that Rune requires a CUDA-capable GPU with sufficient VRAM for the chosen base model (e.g., ~4 GB for a 7B model with QLoRA quantization). Keep the note that QLoRA is used to reduce VRAM requirements.
    - In the "Architecture Overview > Serving Architecture" paragraph (lines 98-109), remove the hardcoded "PP=2" and "two GPUs" language. Rewrite to say the serving layer supports pipeline parallelism across multiple GPUs and can run on a single GPU. Remove the CXL and PCIe-specific commentary. Remove the paragraph about tensor parallelism being excluded.
    - In "Current Status" (lines 159-169), remove references to "Phase 0 validates the hardware environment" and specific hardware. Reframe as Phase 0 validates the software environment (vLLM, PEFT, quantization toolchain).
    - In the "Approach" section (line 53), change "without approaching filesystem or VRAM limits" to just "without approaching filesystem limits" (VRAM depends on hardware).
    - In "Design Principles" (line 127), "Local-first" is fine and should stay. "Sovereign AI" is fine.

    **docs/architecture/multi-gpu-strategy.md:**
    - Rename the document's title to "GPU Strategy" (not "Multi-GPU Strategy").
    - Rewrite the overview to describe Rune's GPU configuration as flexible: single-GPU or multi-GPU. When using multiple GPUs, pipeline parallelism is recommended over tensor parallelism.
    - Remove specific hardware references (2x RTX 4090, CXL, PCIe Gen4 bandwidth numbers, Ada Lovelace sm_89). Replace with general guidance: "For GPUs connected via PCIe (most consumer setups), pipeline parallelism is recommended. Tensor parallelism requires high-bandwidth interconnects like NVLink."
    - Keep the vLLM bug #21471 note as a warning, but frame it as "be aware of this bug if using TP with LoRA" rather than "this is why we forbid TP=2."
    - Rewrite the VRAM Budget section to use a generic example (e.g., "a 7B model with NF4 quantization requires ~4 GB") rather than specific GPU memory allocations per GPU slot.
    - Remove the "Hardware Reference" table at the bottom entirely.
    - Remove the GPU Lease Mechanism section's hardware-specific language. Keep the concept of lease coordination but remove references to "GPU 0" and "GPU 1" specifics — use generic "primary GPU" and "secondary GPU" or just "GPUs."
    - Update the mkdocs.yml nav entry from "Multi-GPU Strategy" to "GPU Strategy" if the filename changes. If keeping the same filename, at least update the nav label.

    **docs/implementation-plan.md:**
    - Phase 0: Remove the "Hardware Constraints" table (lines 60-68). Rewrite Phase 0 as "Environment Validation" — focused on confirming vLLM, PEFT, and the quantization toolchain work correctly. Remove specific GPU specs. The deliverables should be about confirming the software stack works (vLLM serves the model, PEFT fine-tuning runs, etc.) without prescribing specific hardware.
    - Remove the "Note on tensor parallelism" paragraph (line 70).
    - Phase 1: Remove "Hardware note for Phase 1" (line 125) or rewrite without hardware specifics.
    - Phase 2: Remove "PP=2" and "dual RTX 4090" references in the serving infrastructure discussion.
    - Throughout: Replace "24 GB" VRAM references with generic language ("available VRAM"). Replace "both GPUs" with "available GPUs." Replace "PP=2, TP=1" with "pipeline parallelism" (the specific values are config, not docs).

    **docs/architecture/monorepo-mapping.md:**
    - Line 18: Change "vLLM subprocess (PP=2, TP=1, --enable-lora)" to "vLLM subprocess with dynamic LoRA loading"
    - Lines 51-66: Remove "two GPUs" and "GPU lease" specific hardware references. Keep the concept that lora-server and training-svc coordinate GPU access, without prescribing specific GPU counts.

    **docs/architecture/recursive-loop.md:**
    - Line 45: Change "Base SLM + loaded LoRA adapters, served via vLLM PP=2" to "Base SLM + loaded LoRA adapters, served via vLLM"
    - Line 129: Change "vLLM lora-server (Multi-GPU Strategy)" reference to "(GPU Strategy)" if the doc was renamed.

    **docs/article/methods.md:**
    - Lines 13-14: Remove "pipeline parallelism (PP=2)" — just say "vLLM with optional pipeline parallelism"
    - Lines 199-208: Rewrite the "Serving Architecture" subsection. Remove "Rune's hardware is dual RTX 4090 (24 GB VRAM per GPU) without NVLink." Replace with generic: "For multi-GPU setups connected via PCIe, pipeline parallelism is recommended..." Remove specific bandwidth numbers. Keep the conceptual argument.
    - Lines 203-206: Remove specific VRAM budget numbers. Say QLoRA reduces memory requirements to fit consumer GPUs.

    **docs/article/background.md:**
    - Lines 82-85: Remove "For Rune's target hardware (dual RTX 4090, 24 GB per GPU)". Replace with "For consumer GPUs with limited VRAM" or similar generic framing.

    **docs/article/discussion.md:**
    - Line 29: Remove "Phase 0 hardware validation (PP=2 + QLoRA + vLLM compatibility on dual RTX 4090)". Replace with "Phase 0 environment validation (vLLM + QLoRA compatibility)".

    **libs/model-training/README.md:**
    - Line 14: The existing text "If training requires GPU, ensure your environment (Docker/local) passes through GPU resources" is fine — it is generic. No change needed.

    **mkdocs.yml:**
    - Update the nav label from "Multi-GPU Strategy" to "GPU Strategy" to match the renamed doc.

    IMPORTANT: Do NOT remove conceptual technical content. The explanations of WHY pipeline parallelism is preferred over tensor parallelism for PCIe-connected GPUs is valid general knowledge. Keep that reasoning but frame it generically ("GPUs connected via PCIe" not "our two RTX 4090s connected via CXL"). Keep QLoRA rationale but frame it as "consumer GPUs with limited VRAM" not "24 GB RTX 4090."
  </action>
  <verify>
    <automated>cd /Users/noahdolevelixir/Code/rune && ! grep -r "RTX 4090\|Threadripper\|CXL\|CUDA 12.8\|PyTorch 2.9\|48 GB total\|24 GB VRAM\|24 GB per GPU\|sm_89\|Ada Lovelace\|cu128" --include="*.md" --include="*.yml" --include="*.yaml" . | grep -v ".planning/" | grep -v "node_modules/" && echo "PASS: No hardware-specific prescriptions found" || echo "FAIL: Hardware references remain"</automated>
  </verify>
  <done>
    No documentation file prescribes specific GPU hardware (RTX 4090, Threadripper, CXL, specific VRAM amounts, CUDA 12.8, PyTorch 2.9) as a requirement. Hardware choices are presented as examples or recommendations. Technical reasoning about pipeline vs tensor parallelism is preserved but framed generically.
  </done>
</task>

<task type="auto">
  <name>Task 2: Parameterize lora-server config and docker-compose for flexible hardware</name>
  <files>
    services/lora-server/config.py,
    services/lora-server/config.yaml,
    services/lora-server/startup.sh,
    infra/docker-compose.yml
  </files>
  <action>
    Update config and startup files so hardware settings are user-configurable rather than hardcoded.

    **services/lora-server/config.py:**
    - Change `pipeline_parallel_size` default from `2` to `1` (single GPU is the more universal default).
    - Remove the `__post_init__` validation that raises ValueError when `tensor_parallel_size == 2`. Replace with a docstring WARNING note that TP with LoRA may produce corrupted outputs on consumer GPUs (vLLM #21471), but do not forbid it programmatically. Users with NVLink-equipped GPUs should be able to use TP=2.
    - Update the class docstring to remove "Enforces PP=2/TP=1 layout" language. Instead: "Configuration for the vLLM-based LoRA inference server. Default settings target single-GPU operation; adjust pipeline_parallel_size and tensor_parallel_size for multi-GPU setups."
    - Keep `gpu_memory_utilization: float = 0.80` — this is a reasonable default regardless of hardware.

    **services/lora-server/config.yaml:**
    - Change `pipeline_parallel_size` from `2` to `1`.
    - Add comments explaining each setting and how to adjust for multi-GPU.

    **services/lora-server/startup.sh:**
    - Instead of hardcoding `--pipeline-parallel-size 2 --tensor-parallel-size 1`, read these values from config.yaml (or environment variable overrides). The script already reads `model` from config.yaml — extend the same pattern to read `pipeline_parallel_size`, `tensor_parallel_size`, `quantization`, `max_loras`, and `gpu_memory_utilization`.
    - Use environment variable fallbacks: `PIPELINE_PARALLEL_SIZE`, `TENSOR_PARALLEL_SIZE`, etc.
    - Add `--gpu-memory-utilization` flag read from config.

    **infra/docker-compose.yml:**
    - The `deploy.resources.reservations.devices` block with `driver: nvidia, count: all` is fine and should stay — it just asks Docker to pass through whatever GPUs are available. This is already hardware-agnostic. No change needed here.
    - Add a comment above the deploy block explaining that it passes through all available GPUs and can be adjusted.

    Run the existing lora-server tests to confirm nothing breaks:
    ```
    uv run pytest services/lora-server/tests/ -x
    ```
  </action>
  <verify>
    <automated>cd /Users/noahdolevelixir/Code/rune && uv run pytest services/lora-server/tests/ -x -q 2>&1 | tail -5</automated>
  </verify>
  <done>
    - lora-server config.py defaults to single-GPU (PP=1, TP=1) and no longer raises on TP=2
    - config.yaml defaults to PP=1 with explanatory comments
    - startup.sh reads all parallelism settings from config.yaml/env vars instead of hardcoding
    - All existing lora-server tests pass
  </done>
</task>

</tasks>

<verification>
Run a comprehensive grep to confirm no hardware-specific prescriptions remain outside of .planning/:

```bash
# Should return zero matches for prescriptive hardware specs
grep -rn "RTX 4090\|Threadripper\|CXL\|CUDA 12\.8\|PyTorch 2\.9\|48 GB total\|24 GB VRAM\|24 GB per GPU\|sm_89\|Ada Lovelace\|cu128" --include="*.md" --include="*.py" --include="*.yaml" --include="*.yml" --include="*.sh" . | grep -v ".planning/" | grep -v "node_modules/"
```

Confirm lora-server tests pass:
```bash
uv run pytest services/lora-server/tests/ -x
```
</verification>

<success_criteria>
1. Zero occurrences of specific hardware model names (RTX 4090, Threadripper, CXL interconnect) as requirements in any doc or config file outside .planning/
2. README.md Hardware section describes the system as running on any local CUDA-capable GPU setup
3. lora-server defaults to single-GPU operation (PP=1) and startup.sh reads config from YAML/env vars
4. All existing tests pass
5. Technical reasoning (pipeline vs tensor parallelism, QLoRA rationale) is preserved but framed generically
</success_criteria>

<output>
After completion, create `.planning/quick/1-remove-all-hardware-requirements-make-re/1-SUMMARY.md`
</output>
