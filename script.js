document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('calculator-form');
    const taskTypeSelect = document.getElementById('task-type');
    const precisionSelect = document.getElementById('precision');
    const modelSizeInput = document.getElementById('model-size');
    const batchSizeInput = document = document.getElementById('batch-size');
    const seqLenInput = document.getElementById('seq-len');
    const toggleAdvancedBtn = document.getElementById('toggle-advanced');
    const advancedSettings = document.getElementById('advanced-settings');

    // Dynamic Form Groups
    const loraRankGroup = document.getElementById('lora-rank-group');
    const loraAlphaGroup = document.getElementById('lora-alpha-group');
    const trainableParamsGroup = document.getElementById('trainable-params-group');
    const optimizerGroup = document.getElementById('optimizer-group');
    const kvCacheRow = document.getElementById('kv-cache-row');
    const gradRow = document.getElementById('grad-row');
    const optRow = document.getElementById('opt-row');

    // Time Estimation Inputs
    const datasetGroup = document.getElementById('dataset-group');
    const epochsGroup = document.getElementById('epochs-group');
    const outputTokensGroup = document.getElementById('output-tokens-group');
    const datasetTokensInput = document.getElementById('dataset-tokens');
    const epochsInput = document.getElementById('epochs');
    const outputTokensInput = document.getElementById('output-tokens');

    // Result Elements
    const totalVramDisplay = document.getElementById('total-vram-display');
    const weightMemoryDisplay = document.getElementById('weight-memory');
    const kvMemoryDisplay = document.getElementById('kv-memory');
    const gradMemoryDisplay = document.getElementById('grad-memory');
    const optMemoryDisplay = document.getElementById('opt-memory');
    const actMemoryDisplay = document.getElementById('act-memory');
    const overheadMemoryDisplay = document.getElementById('overhead-memory');
    const timeEstimateDisplay = document.getElementById('time-estimate');
    const gpuRecommendations = document.getElementById('gpu-recommendations');

    // Constants
    const BYTES_PER_GB = 1024 * 1024 * 1024;

    // GPU Database with TFLOPS (FP16/BF16 Tensor) and Bandwidth (GB/s)
    // TFLOPS are approximate peak Tensor Core performance (often with sparsity, but we use dense or a factor)
    // We'll use a conservative "effective" TFLOPS for estimation (e.g. 50% of peak dense)
    const GPUS = [
        { name: 'NVIDIA H100', vram: 80, tflops: 1000, bw: 3350 }, // ~2000 dense, use 1000 effective
        { name: 'NVIDIA A100', vram: 80, tflops: 312, bw: 2039 }, // 312 dense
        { name: 'NVIDIA A100', vram: 40, tflops: 312, bw: 1555 },
        { name: 'NVIDIA A6000', vram: 48, tflops: 150, bw: 768 }, // Ampere
        { name: 'NVIDIA RTX 6000 Ada', vram: 48, tflops: 300, bw: 960 }, // Ada
        { name: 'NVIDIA RTX 4090', vram: 24, tflops: 165, bw: 1008 }, // ~165 dense TC
        { name: 'NVIDIA RTX 3090/Ti', vram: 24, tflops: 71, bw: 936 }, // ~71 dense TC
        { name: 'NVIDIA L4', vram: 24, tflops: 120, bw: 300 }, // Ada
        { name: 'NVIDIA A10G', vram: 24, tflops: 70, bw: 600 },
        { name: 'NVIDIA RTX 4080', vram: 16, tflops: 97, bw: 717 },
        { name: 'NVIDIA T4', vram: 16, tflops: 65, bw: 320 },
        { name: 'NVIDIA RTX 3080', vram: 10, tflops: 58, bw: 760 },
    ];

    // Event Listeners
    form.addEventListener('input', calculateMemory);
    taskTypeSelect.addEventListener('change', updateFormVisibility);
    toggleAdvancedBtn.addEventListener('click', () => {
        advancedSettings.classList.toggle('hidden');
        toggleAdvancedBtn.textContent = advancedSettings.classList.contains('hidden')
            ? 'Show Advanced Settings'
            : 'Hide Advanced Settings';
    });

    // Initial Setup
    updateFormVisibility();
    calculateMemory();

    function updateFormVisibility() {
        const task = taskTypeSelect.value;
        const isTraining = task.includes('training');
        const isLora = task.includes('lora');

        // Toggle Visibility
        loraRankGroup.style.display = isLora ? 'block' : 'none';
        loraAlphaGroup.style.display = isLora ? 'block' : 'none';
        trainableParamsGroup.style.display = isLora ? 'block' : 'none';
        optimizerGroup.style.display = isTraining ? 'block' : 'none';

        // Time Estimation Inputs
        datasetGroup.style.display = isTraining ? 'block' : 'none';
        epochsGroup.style.display = isTraining ? 'block' : 'none';
        outputTokensGroup.style.display = task === 'inference' ? 'block' : 'none';

        // Result Rows Visibility
        kvCacheRow.style.display = task === 'inference' ? 'flex' : 'none';
        gradRow.style.display = isTraining ? 'flex' : 'none';
        optRow.style.display = isTraining ? 'flex' : 'none';

        // Auto-select optimizer defaults
        if (task === 'training_qlora') {
            document.getElementById('optimizer').value = 'paged_adamw_8bit';
        } else if (task === 'training_lora') {
            document.getElementById('optimizer').value = 'adamw_8bit';
        } else if (task === 'training_full') {
            document.getElementById('optimizer').value = 'adamw';
        }

        calculateMemory();
    }

    function getBytesPerParam(precision) {
        switch (precision) {
            case 'fp32': return 4;
            case 'fp16': return 2; // includes bf16
            case 'int8': return 1;
            case 'int4': return 0.5;
            default: return 2;
        }
    }

    function calculateMemory() {
        // Inputs
        const task = taskTypeSelect.value;
        const modelSizeB = parseFloat(modelSizeInput.value) || 0; // Billions
        const modelParams = modelSizeB * 1_000_000_000;
        const precision = precisionSelect.value;
        const batchSize = parseInt(batchSizeInput.value) || 1;
        const seqLen = parseInt(seqLenInput.value) || 2048;

        // Advanced Inputs
        const numLayers = parseInt(document.getElementById('num-layers').value) || 32;
        const numHeads = parseInt(document.getElementById('num-heads').value) || 32;
        const numKVHeads = parseInt(document.getElementById('num-kv-heads').value) || numHeads; // Default to numHeads if not set
        const hiddenDim = parseInt(document.getElementById('hidden-dim').value) || 4096;
        const loraRank = parseInt(document.getElementById('lora-rank').value) || 64;
        const trainablePercent = parseFloat(document.getElementById('trainable-percent').value) || 0.2; // %

        let weightMem = 0;
        let kvMem = 0;
        let gradMem = 0;
        let optMem = 0;
        let actMem = 0;

        // 1. Model Weights
        const bytesPerParam = getBytesPerParam(precision);
        weightMem = modelParams * bytesPerParam;

        // 2. KV Cache (Inference)
        // Formula: 2 * Batch * SeqLen * Layers * KV_Heads * HeadDim * Precision
        // HeadDim = HiddenDim / Attn_Heads
        const headDim = hiddenDim / numHeads;
        const kvPrecisionBytes = 2; // Usually FP16/BF16

        // Note: Even in training, KV cache is used during the forward pass if not using gradient checkpointing,
        // but typically activation memory dominates. For inference, it's explicit.
        if (task === 'inference') {
            kvMem = 2 * batchSize * seqLen * numLayers * numKVHeads * headDim * kvPrecisionBytes;
        }

        // 3. Training Components
        if (task.includes('training')) {
            const optimizerType = document.getElementById('optimizer').value;
            let optBytesPerParam = 12; // AdamW default
            if (optimizerType === 'sgd') optBytesPerParam = 4;
            if (optimizerType.includes('8bit')) optBytesPerParam = 6;

            if (task === 'training_full') {
                gradMem = modelParams * 4; // FP32 gradients
                optMem = modelParams * optBytesPerParam;

                // Activation Memory (Refined)
                // A better approximation for Transformer activations (with recomputation/checkpointing off):
                // Act ≈ Batch * SeqLen * HiddenDim * Layers * (34 + (5 * SeqLen * AttnHeads) / (HiddenDim)) ... complex
                // Simplified "safe" upper bound often used:
                // Act ≈ Batch * SeqLen * HiddenDim * Layers * 12 (bytes) for FP16 training without checkpointing
                // With Gradient Checkpointing (Activation Checkpointing), it drops to ~ sqrt(Layers) or just storing inputs.
                // Let's assume Gradient Checkpointing is ON for large models (standard practice).
                // Act ≈ 2 * Batch * SeqLen * HiddenDim * Layers * 2 bytes (very rough, but standard "checkpointing" cost is lower)
                // Let's use a conservative estimate for "modern" training with checkpointing:
                // ~ 2 GB per billion params for small batch/seq? No, that's weights.

                // Let's use the formula from the research:
                // Activations ≈ Batch * SeqLen * HiddenDim * (34 + ...)
                // Let's stick to a simpler heuristic that scales:
                // Act = Batch * SeqLen * HiddenDim * Layers * 2 (bytes) * 2 (safety factor)
                actMem = batchSize * seqLen * numLayers * hiddenDim * 4;
            }
            else if (task.includes('lora')) {
                const trainableParams = modelParams * (trainablePercent / 100);
                gradMem = trainableParams * 4;
                optMem = trainableParams * optBytesPerParam;

                // Activations for LoRA/QLoRA
                // Similar to full training but often with gradient checkpointing enabled by default in libraries like PEFT.
                // It's usually significantly lower than full fine-tuning without checkpointing.
                actMem = batchSize * seqLen * numLayers * hiddenDim * 2;
            }
        }

        // 4. Activations (Inference)
        if (task === 'inference') {
            // Inference activations: Batch * SeqLen * HiddenDim + overheads
            // Usually quite small compared to KV cache for long sequences
            actMem = batchSize * seqLen * hiddenDim * 2;
        }

        // Total
        let totalMem = weightMem + kvMem + gradMem + optMem + actMem;

        // Overhead (CUDA kernels, fragmentation)
        // 10-20% is standard. Let's use 15%.
        const overheadMem = totalMem * 0.15;
        const finalTotalMem = totalMem + overheadMem;

        // Update Display
        updateDisplay(weightMem, kvMem, gradMem, optMem, actMem, overheadMem, finalTotalMem);

        // Recommend GPUs and Estimate Time
        const recommended = recommendGPUs(finalTotalMem);

        // Estimate Time based on the best single GPU found, or the first valid config
        if (recommended && recommended.length > 0) {
            estimateTime(recommended[0], task, modelParams, precision);
        } else {
            timeEstimateDisplay.textContent = "N/A (No suitable GPU)";
        }
    }

    function updateDisplay(w, k, g, o, a, ov, total) {
        const toGB = (bytes) => (bytes / BYTES_PER_GB).toFixed(2) + ' GB';

        weightMemoryDisplay.textContent = toGB(w);
        kvMemoryDisplay.textContent = toGB(k);
        gradMemoryDisplay.textContent = toGB(g);
        optMemoryDisplay.textContent = toGB(o);
        actMemoryDisplay.textContent = toGB(a);
        overheadMemoryDisplay.textContent = toGB(ov);
        totalVramDisplay.textContent = toGB(total);

        // Add unit label
        if (!totalVramDisplay.innerHTML.includes('span')) {
            totalVramDisplay.innerHTML += ' <span>Total VRAM</span>';
        }
    }

    function recommendGPUs(totalBytes) {
        const totalGB = totalBytes / BYTES_PER_GB;
        gpuRecommendations.innerHTML = '';

        const validConfigs = [];

        // Single GPU
        GPUS.forEach(gpu => {
            if (gpu.vram >= totalGB) {
                validConfigs.push({ name: gpu.name, count: 1, vram: gpu.vram, tflops: gpu.tflops, bw: gpu.bw });
            }
        });

        // Multi GPU (2x, 4x, 8x)
        [2, 4, 8].forEach(count => {
            if (validConfigs.length < 5) { // Only search if we need more options
                GPUS.forEach(gpu => {
                    if (gpu.vram * count >= totalGB) {
                        // Check if we already have this GPU with a lower count (optimization)
                        // Actually we want to show 2x 3090 even if 1x A100 works, because it's cheaper.
                        validConfigs.push({ name: gpu.name, count: count, vram: gpu.vram * count, tflops: gpu.tflops, bw: gpu.bw });
                    }
                });
            }
        });

        // Sort: Primary by Count (fewer is better), Secondary by VRAM (closer fit is better)
        validConfigs.sort((a, b) => {
            if (a.count !== b.count) return a.count - b.count;
            return a.vram - b.vram;
        });

        // Deduplicate (e.g. don't show 2x A100 if 1x A100 is in list? No, keep explicit)
        // But maybe filter out "2x H100" if "1x H100" is there.
        const uniqueConfigs = [];
        const seen = new Set();
        validConfigs.forEach(c => {
            const key = `${c.name}-${c.count}`; // unique config
            // Heuristic: if we have 1x GPU, don't show 2x of SAME GPU.
            // But do show 2x of WEAKER GPU.
            const singleVersion = validConfigs.find(v => v.name === c.name && v.count === 1);
            if (c.count > 1 && singleVersion) return;

            if (!seen.has(key)) {
                uniqueConfigs.push(c);
                seen.add(key);
            }
        });

        if (uniqueConfigs.length === 0) {
            gpuRecommendations.innerHTML = '<div class="placeholder-text">No standard configuration found. Requires cluster.</div>';
            return [];
        }

        uniqueConfigs.slice(0, 5).forEach(config => {
            const div = document.createElement('div');
            div.className = 'gpu-item';
            if (config.count === 1 && config.name.includes('4090')) div.classList.add('recommended'); // Highlight consumer favorite

            div.innerHTML = `
                <div>
                    <div class="gpu-name">${config.name}</div>
                    <div class="gpu-vram">Total VRAM: ${config.vram} GB</div>
                </div>
                <div class="gpu-count">${config.count}x</div>
            `;
            gpuRecommendations.appendChild(div);
        });

        return uniqueConfigs;
    }

    function estimateTime(config, task, modelParams, precision) {
        // config: { name, count, tflops, bw }
        // tflops is FP16 Tensor Core effective

        let timeStr = "";

        if (task === 'inference') {
            // Inference Time Estimation
            // Dominated by memory bandwidth for decoding (generation)
            // Dominated by compute for prefill (prompt processing)

            const outputTokens = parseInt(outputTokensInput.value) || 512;
            const batchSize = parseInt(batchSizeInput.value) || 1;

            // Bandwidth in GB/s -> Bytes/s
            const totalBW = config.bw * config.count * 1_000_000_000;

            // Model Size in Bytes (for bandwidth bound)
            // Even if INT4, we load INT4 weights.
            const bytesPerParam = getBytesPerParam(precision);
            const modelBytes = modelParams * bytesPerParam;

            // Time per token (Generation) = ModelBytes / Bandwidth
            // This is for batch size 1 (or small batches where we are bandwidth bound)
            // For large batches, we might become compute bound.
            // Let's assume bandwidth bound for typical interactive use.
            const timePerToken = modelBytes / totalBW;

            // Total Generation Time
            const genTime = timePerToken * outputTokens;

            // Tokens per Second
            const tps = 1 / timePerToken;

            timeStr = `~${tps.toFixed(1)} tokens/sec (Gen)`;

        } else {
            // Training Time Estimation
            // Formula: 6 * Params * Tokens * Epochs / (Compute * Efficiency)

            const datasetTokensB = parseFloat(datasetTokensInput.value) || 1;
            const datasetTokens = datasetTokensB * 1_000_000_000;
            const epochs = parseInt(epochsInput.value) || 1;

            // Total Compute Needed (FLOPs)
            // Full Finetune: 6 FLOPs per param per token
            // LoRA: Forward is full cost, Backward is partial.
            // Forward ~ 2 FLOPs/param. Backward ~ 4 FLOPs/param (Full).
            // LoRA Backward: Gradients for adapters only, but need to compute activations?
            // LoRA is often ~70-80% of full training cost per step, or sometimes same speed but less memory.
            // Let's use a factor.
            let flopFactor = 6;
            if (task.includes('lora')) {
                flopFactor = 4; // Rough estimate: Forward (2) + Backward (Adapters + Activation Recompute overhead) ~ 4?
            }

            const totalFlops = flopFactor * modelParams * datasetTokens * epochs;

            // Total Compute Available (FLOPS)
            // TFLOPS -> FLOPS
            // Efficiency (MFU): 30% - 50% is typical. Let's use 40%.
            const efficiency = 0.4;
            const totalComputePerSec = config.tflops * 1_000_000_000_000 * config.count * efficiency;

            const totalSeconds = totalFlops / totalComputePerSec;

            // Format time
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);

            if (hours > 24) {
                const days = (hours / 24).toFixed(1);
                timeStr = `~${days} days`;
            } else if (hours > 0) {
                timeStr = `~${hours}h ${minutes}m`;
            } else {
                timeStr = `~${minutes} min`;
            }
        }

        timeEstimateDisplay.textContent = timeStr + ` (on ${config.count}x ${config.name})`;
    }
});
