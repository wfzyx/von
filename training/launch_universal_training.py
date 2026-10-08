#!/usr/bin/env python3
"""Launch Phase 4 Universal Decision Corpus Training on AWS Spot GPUs.

Targets 4x NVIDIA T4 (g4dn.12xlarge) running 200,000 samples across all 49
operational decision domains.
"""

import json
import os
import shutil
import subprocess
import time

VENV_AWS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "aws"))
AWS_CLI = VENV_AWS if os.path.exists(VENV_AWS) else (shutil.which("aws") or "/mnt/c/Program Files/Amazon/AWSCLIV2/aws.exe")
REGION = "us-west-2"
SUBNETS = [
    ("subnet-083ba040", "us-west-2a"),
    ("subnet-070e7461", "us-west-2b"),
    ("subnet-b75369ec", "us-west-2c"),
    ("subnet-a663228e", "us-west-2d"),
]
AMI_ID = "ami-0e24e0019a12c5b13"

CANDIDATE_TYPES = [
    # A10G first: ~2x the throughput of a T4 for this workload and 24GB per GPU,
    # which makes it both faster and cheaper per epoch despite the higher rate.
    ("g5.12xlarge", "4x NVIDIA A10G 96GB, 48 vCPU (On-Demand ~$5.67/hr)"),
    # Same 24 GB/GPU class, separate capacity pool; ~0.8x A10G throughput.
    ("g6.12xlarge", "4x NVIDIA L4 96GB, 48 vCPU (On-Demand ~$4.60/hr)"),
    # 48 GB/GPU: takes batch 8 on long rows; pricier but rarely sold out.
    ("g6e.12xlarge", "4x NVIDIA L40S 192GB, 48 vCPU (On-Demand ~$10.50/hr)"),
    # g4dn (16 GB T4) OOMs on ~1.2k-token judge rows at batch 8; A10G only.
    ("g5.4xlarge", "1x NVIDIA A10G 24GB, 16 vCPU (Spot ~$0.69/hr)"),
    ("g5.2xlarge", "1x NVIDIA A10G 24GB, 8 vCPU (Spot ~$0.54/hr)"),
]
IAM_PROFILE = "AmazonSSMRoleForInstancesQuickSetup"
S3_TARGET = "s3://model-weight/von-option-marker-universal"

MARKER_TRAIN_BLOCK = """# Detect GPUs and train with DDP
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs. Starting PyTorch DDP training with 8,192 Context Window..."

/opt/von/.venv/bin/torchrun --nproc_per_node=$NUM_GPUS training/train_option_marker.py \\
    --train_data data_universal/train.jsonl \\
    --val_data data_universal/val.jsonl \\
    --base_model_id {base_model_id} \\
    {init_ckpt_flag} \\
    {independent_options_flag} \\
    --epochs {epochs} \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --grad_accum_steps {grad_accum} \\
    --max_position_embeddings 8192 \\
    --long_ratio {long_ratio} \\
    --s3_target {s3_target} \\
    --output_dir checkpoints/von-long-context

"""

DECODER_TRAIN_BLOCK = """# Decoder variant: Qwen3.5-0.8B + LoRA + JointSchemaHead (training/train_decoder_head.py)
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs. Starting decoder+head DDP training..."
/opt/von/.venv/bin/torchrun --nproc_per_node=$NUM_GPUS training/train_decoder_head.py \\
    --train_data data_universal/train.jsonl \\
    --val_data data_universal/val.jsonl \\
    --base_model_id {base_model_id} \\
    --epochs {epochs} \\
    --lr {lr} \\
    --batch_size {batch_size} --max_tokens {max_tokens} \\
    --grad_accum_steps {grad_accum} \\
    --max_length {max_length} \\
    --lora_r {lora_r} --lora_alpha {lora_alpha} \\
    --head_width {head_width} --head_routing_layers {head_routing_layers} --head_layers {head_layers} \\
    --head_heads {head_heads} --head_feedforward {head_feedforward} {gc_flag} \\
    --polarity_weight {polarity_weight} --polarity_margin {polarity_margin} \\
    --s3_target {s3_target} \\
    --output_dir checkpoints/von-2-nano

# Gate eval on the held-out suites (Von-1.2 and Jeff-0.8B dumps live in benchmarks/data/gate_cache for comparison)
mkdir -p /opt/von/jevbench-public
aws s3 sync s3://model-weight/jevbench-public/ /opt/von/jevbench-public/ --only-show-errors
export JEVBENCH_PUBLIC=/opt/von/jevbench-public
/opt/von/.venv/bin/python training/eval_decoder.py --checkpoint checkpoints/von-2-nano --name {eval_name} \\
    --suites judge_heldout,probes,jabr_v2,jev_easy,jev_standard,jev_hard --out_dir /opt/von/gate_out || true
aws s3 cp /opt/von/gate_out/{eval_name}.json {s3_target}/gate_cache/{eval_name}.json || true
"""

USER_DATA_TEMPLATE = """#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

# Any failure must stop billing. Without this, `set -e` exits before the
# shutdown line below and the instance idles until the watchdog fires.
cleanup() {{
  rc=$?
  echo "=== [EXIT rc=$rc] uploading log and shutting down ==="
  aws s3 cp /var/log/user-data.log {s3_target}/run.log || true
  shutdown -h now
}}
trap cleanup EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cancel any lingering shutdown timer and set a hard watchdog. Sized from the
# run's own epoch estimate (see launch()), not a fixed number: a 240-minute
# ceiling killed a 2-epoch full-corpus run 23 minutes before it finished.
shutdown -c 2>/dev/null || true
# Timer-driven shutdown skips the EXIT trap, so the watchdog ships the log itself.
( sleep {watchdog_min}m; echo "=== WATCHDOG {watchdog_min}m: halting ==="; aws s3 cp /var/log/user-data.log {s3_target}/run.log || true; shutdown -h now ) &

echo "=== [VON UNIVERSAL DECISION TRAINING START] ==="
export DEBIAN_FRONTEND=noninteractive

# Ubuntu's unattended-upgrades grabs the dpkg lock during boot and races this
# install; losing that race exits the whole script (rc=100) after the instance
# is already billing. Stop the timer, then still pass a lock timeout in case it
# is mid-run.
systemctl stop unattended-upgrades.service 2>/dev/null || true
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
for i in $(seq 1 60); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  echo "waiting for dpkg lock ($i)..."; sleep 5
done
APT_OPTS="-o DPkg::Lock::Timeout=600 -y"
apt-get $APT_OPTS update && apt-get $APT_OPTS install awscli curl git

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

mkdir -p /opt/von
aws s3 cp s3://model-weight/von-marker-src.tar.gz /tmp/von-marker-src.tar.gz
tar -xzf /tmp/von-marker-src.tar.gz -C /opt/von
cd /opt/von

# Pin the interpreter to the one uv.lock targets. An unpinned venv picked CPython 3.14 and, on 2026-10-07, the
# solver backtracked `datasets` to 1.1.1 (2020) which imports pyarrow.PyExtensionType (removed) -> r3 died at corpus
# build. r2 five days earlier had resolved datasets 5.0.1 on the same command.
/root/.local/bin/uv venv --clear --python 3.12 /opt/von/.venv
# PyPI ships CUDA-enabled Linux torch wheels; the old cu121 index now 404s and
# breaks the solve. One install so torch and its dependents resolve together.
/root/.local/bin/uv pip install --python /opt/von/.venv torch torchvision transformers 'datasets>=5,<6' 'pyarrow<26' scipy sentencepiece tiktoken accelerate pydantic awscli {extra_pip}
/opt/von/.venv/bin/python -c 'import datasets, pyarrow; print("datasets", datasets.__version__, "pyarrow", pyarrow.__version__)'
export PYTHONPATH="/opt/von/src:$PYTHONPATH"

# Build the Universal Decision Corpus, including the long-context core
echo "=== Building {max_train}-sample Universal Decision Corpus (long_context={long_context}, synthetic={synthetic_n}) ==="
/opt/von/.venv/bin/python -m training.prepare_universal_dataset \\
    --max_train {max_train} --val_samples 5000 \\
    --long_context {long_context} --overlap_target {overlap_target} \\
    --synthetic_n {synthetic_n} \\
    --output_dir data_universal

{extra_train_block}
# Optional: continue from an existing full checkpoint (encoder + trained scoring
# head) instead of a randomly-initialised head on the base encoder.
{init_ckpt_block}

{train_block}
aws s3 cp /var/log/user-data.log {s3_target}/run.log || true

echo "=== [UNIVERSAL TRAINING COMPLETE - TERMINATING] ==="
shutdown -h now
"""


def run_aws(cmd: list) -> dict:
    full_cmd = [AWS_CLI] + cmd + ["--region", REGION, "--output", "json"]
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"AWS CLI error: {res.stderr.strip()}")
    if not res.stdout.strip():
        return {}
    return json.loads(res.stdout)


def launch(
    on_demand: bool = False,
    epochs: int = 3,
    s3_target: str = S3_TARGET,
    max_train: int = 290000,
    long_context: int = 40000,
    long_ratio: float = 0.30,
    overlap_target: float = 0.32,
    synthetic_n: int = 0,
    extra_train_s3: str = "",
    batch_size: int = 8,
    grad_accum: int = 2,
    only_type: str = "",
    region: str = "",
    init_checkpoint_s3: str = "",
    independent_options: bool = False,
    base_model_id: str = "wfzyx/von",
    base_encoder_s3: str = "",
    lr: float = 3e-5,
    watchdog_min: int = 0,
    trainer: str = "marker",
    decoder_opts: dict | None = None,
    extra_rows: int = 0,
):
    if trainer not in ("marker", "decoder"):
        raise SystemExit(f"--trainer must be marker or decoder, got {trainer!r}")
    if trainer == "decoder" and (init_checkpoint_s3 or base_encoder_s3 or independent_options):
        raise SystemExit("--trainer decoder takes --base-model-id (a Qwen3.5 Hub id); init/base-encoder/independent-options are marker-only")
    if init_checkpoint_s3 and base_encoder_s3:
        raise SystemExit("--init-checkpoint-s3 and --base-encoder-s3 are exclusive: an init checkpoint carries its own encoder")
    market_str = "On-Demand (Guaranteed)" if on_demand else "Spot"
    print("================================================================")
    print(f"  VON TRAINING LAUNCHER [{market_str}]")
    print(f"  Corpus:           {max_train:,} samples (+{long_context:,} long-context)")
    print(f"  Epochs:           {epochs}")
    print(f"  Long batch ratio: {long_ratio:.0%}")
    print(f"  Overlap target:   {overlap_target:.0%} gold-is-highest-overlap")
    print(f"  Synthetic rows:   {synthetic_n:,}")
    print(f"  Init checkpoint:  {init_checkpoint_s3 or '(none - fresh scoring head)'}")
    print(f"  Trainer:          {trainer}")
    print(f"  Base encoder:     {base_encoder_s3 or base_model_id}")
    print(f"  Independent opts: {independent_options}")
    print(f"  LR:               {lr}")
    # Full-corpus epochs measured at ~110 min on 4x T4 with 8192-token rows; budget
    # 130 min/epoch plus 40 min for corpus build + sync, unless overridden.
    if watchdog_min <= 0:
        if trainer == "decoder":
            # Measured on 4x A10G: 18.6 rows/s on the universal corpus, 11 rows/s once judge/RAGTruth-length rows
            # are mixed in. Budget the slow rate over every row the run will see, plus setup and eval.
            watchdog_min = 40 + int(epochs * (max_train + long_context + synthetic_n + extra_rows) / 11 / 60) + 30
        else:
            watchdog_min = 40 + 130 * epochs
    print(f"  Watchdog:         {watchdog_min} min")
    print("  Cluster Target:   4x GPU (g4dn.12xlarge / g5.12xlarge)")
    print("  Region:           us-west-2")
    print(f"  Target S3 Prefix: {s3_target}")
    if s3_target == S3_TARGET:
        print("  !! WARNING: writing to the SHIPPED weights prefix.")
    print("================================================================\n")

    extra_train_block = ""
    if extra_train_s3:
        # Extra pre-built rows (e.g. data_judge_mix) appended to the built corpus and reshuffled.
        extra_train_block = (
            f"aws s3 cp {extra_train_s3} /opt/von/extra_train.jsonl\n"
            "/opt/von/.venv/bin/python - <<'PYEOF'\n"
            "import random\n"
            "# split on newline only: str.splitlines() also breaks on U+2028/U+2029 inside JSON strings\n"
            "rows = [r for f in ('data_universal/train.jsonl', '/opt/von/extra_train.jsonl') for r in open(f, encoding='utf-8').read().split('\\n') if r.strip()]\n"
            "random.Random(0).shuffle(rows)\n"
            "open('data_universal/train.jsonl', 'w').write('\\n'.join(rows) + '\\n')\n"
            "print('train rows after extra:', len(rows))\n"
            "PYEOF"
        )
    user_data_path = "/tmp/user_data_universal.sh"
    with open(user_data_path, "w") as f:
        if init_checkpoint_s3:
            init_ckpt_block = (
                "mkdir -p /opt/von/init_ckpt\n"
                f"aws s3 sync {init_checkpoint_s3}/ /opt/von/init_ckpt/ --exclude 'run.log'\n"
                "echo '=== init checkpoint downloaded ==='"
            )
            init_ckpt_flag = "--init_checkpoint /opt/von/init_ckpt"
            # The init checkpoint dir also carries the encoder config the model
            # needs, so point base_model_id there rather than at the Hub.
            base_model_id = "/opt/von/init_ckpt"
        elif base_encoder_s3:
            # A plain encoder directory (config + safetensors + tokenizer), e.g. the
            # output of training/extract_gliclass_encoder.py. Fresh scoring head.
            init_ckpt_block = (
                "mkdir -p /opt/von/base_enc\n"
                f"aws s3 sync {base_encoder_s3}/ /opt/von/base_enc/\n"
                "echo '=== base encoder downloaded: fresh scoring head ==='"
            )
            init_ckpt_flag = ""
            base_model_id = "/opt/von/base_enc"
        else:
            init_ckpt_block = "echo '=== no init checkpoint: fresh scoring head on base encoder ==='"
            init_ckpt_flag = ""
        if trainer == "decoder":
            d = dict(max_length=4096, lora_r=64, lora_alpha=128, head_width=512, head_routing_layers=1, head_layers=2,
                     head_heads=8, head_feedforward=2048, eval_name="von-2-nano", gradient_checkpointing=1, max_tokens=16384,
                     polarity_weight=0.0, polarity_margin=2.0)
            d.update(decoder_opts or {})
            # Measured on 24 GB GPUs: checkpointing + 4x4096-token batches = 6.7 GB and stable; without
            # checkpointing both a 32k-token bucket and a 10k-token budget OOM'd in the deltanet backward.
            # So: checkpointing on, 16k padded-token budget, shapes bucketed for Triton autotune.
            d["gc_flag"] = "--gradient_checkpointing" if int(d.pop("gradient_checkpointing")) else ""
            train_block = DECODER_TRAIN_BLOCK.format(base_model_id=base_model_id, epochs=epochs, lr=lr, batch_size=batch_size,
                                                     grad_accum=grad_accum, s3_target=s3_target, **d)
            extra_pip = ("peft safetensors flash-linear-attention && "
                         "CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=24 /root/.local/bin/uv pip install --python /opt/von/.venv "
                         "--no-build-isolation causal-conv1d || echo 'causal-conv1d build failed; reference kernel'")
        else:
            train_block = MARKER_TRAIN_BLOCK.format(base_model_id=base_model_id, init_ckpt_flag=init_ckpt_flag,
                                                    independent_options_flag="--independent_options" if independent_options else "",
                                                    epochs=epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum,
                                                    long_ratio=long_ratio, s3_target=s3_target)
            extra_pip = ""
        f.write(USER_DATA_TEMPLATE.format(
            train_block=train_block,
            extra_pip=extra_pip,
            epochs=epochs,
            s3_target=s3_target,
            max_train=max_train,
            long_context=long_context,
            overlap_target=overlap_target,
            synthetic_n=synthetic_n,
            extra_train_block=extra_train_block,
            init_ckpt_block=init_ckpt_block,
            watchdog_min=watchdog_min,
        ))

    instance_id = None
    selected_type = None

    global REGION, SUBNETS, AMI_ID
    if region and region != REGION:
        # Other regions: default-VPC subnets and the same-name DL base AMI resolved live.
        REGION = region
        SUBNETS = [(sn["SubnetId"], sn["AvailabilityZone"]) for sn in sorted(
            run_aws(["ec2", "describe-subnets", "--filters", "Name=default-for-az,Values=true",
                     "--query", "Subnets[].{SubnetId:SubnetId,AvailabilityZone:AvailabilityZone}"]) or [],
            key=lambda x: x["AvailabilityZone"])]
        imgs = run_aws(["ec2", "describe-images", "--owners", "amazon",
                        "--filters", "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*",
                        "--query", "sort_by(Images,&CreationDate)[-1].ImageId"])
        if not imgs or not SUBNETS:
            raise RuntimeError(f"could not resolve AMI/subnets in {REGION}")
        AMI_ID = imgs
        print(f"  Region {REGION}: AMI {AMI_ID}, subnets {[az for _, az in SUBNETS]}")
    for itype, desc in CANDIDATE_TYPES:
        if only_type and itype not in only_type.split(","):
            continue
        print(f"\nEvaluating instance type: {itype} [{desc}]...")
        offered = set(run_aws(["ec2", "describe-instance-type-offerings", "--location-type", "availability-zone",
                               "--filters", f"Name=instance-type,Values={itype}",
                               "--query", "InstanceTypeOfferings[].Location"]) or [])
        for subnet_id, az in SUBNETS:
            if offered and az not in offered:
                print(f"  -> {itype} not offered in {az}, skipping.")
                continue
            print(f"  -> Trying {itype} in {az} ({subnet_id})...")
            try:
                run_args = [
                    "ec2", "run-instances",
                    "--image-id", AMI_ID,
                    "--instance-type", itype,
                    "--subnet-id", subnet_id,
                    "--iam-instance-profile", f"Name={IAM_PROFILE}",
                    "--user-data", f"file://{user_data_path}",
                    "--count", "1",
                    "--tag-specifications", json.dumps([{
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": f"von-universal-phase4-{'ondemand' if on_demand else 'spot'}"}]
                    }]),
                ]
                if not on_demand:
                    run_args.extend(["--instance-market-options", json.dumps({"MarketType": "spot"})])

                res = run_aws(run_args)
                instance_id = res["Instances"][0]["InstanceId"]
                selected_type = itype
                print(f"\n-> SUCCESS! Launched {itype} {market_str} instance in {az}: {instance_id}")
                break
            except Exception as e:
                err = str(e)
                if "InsufficientInstanceCapacity" in err or "Unsupported" in err or "SpotMaxPriceTooLow" in err:
                    print(f"     Capacity unavailable in {az}.")
                    continue
                print(f"     Failed: {err}")
        if instance_id:
            break

    if not instance_id:
        print(f"\nAll {market_str} candidate pools exhausted.")
        return

    print("\nWaiting for instance to enter 'running' state...")
    while True:
        desc = run_aws(["ec2", "describe-instances", "--instance-ids", instance_id])
        state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"  -> Instance {instance_id} status: {state}")
        if state == "running":
            break
        time.sleep(10)

    print(f"\nUniversal Phase 4 {market_str} training instance is RUNNING!")
    print(f"Artifacts will automatically upload to {s3_target} upon completion.")
    return instance_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Von training on AWS GPUs.")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use guaranteed On-Demand capacity instead of Spot.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=290000)
    parser.add_argument("--long-context", type=int, default=40000)
    parser.add_argument("--overlap-target", type=float, default=0.32,
                        help="target share of items where gold is the highest-overlap option")
    parser.add_argument("--long-ratio", type=float, default=0.30)
    parser.add_argument("--s3-target", type=str, default=S3_TARGET,
                        help="S3 prefix for checkpoints. Defaults to the SHIPPED weights "
                             "prefix, so point experiments somewhere else.")
    parser.add_argument("--region", default="", help="launch in another region (default us-west-2); S3 stays in us-west-2")
    parser.add_argument("--only-type", default="", help="comma-separated allowed instance types, e.g. g5.12xlarge,g6.12xlarge (no single-GPU fallback)")
    parser.add_argument("--batch-size", type=int, default=8, help="per-GPU micro-batch; 4 when the corpus has ~1k-token rows")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--extra-train-s3", default="", help="S3 jsonl of extra rows appended to the built corpus")
    parser.add_argument("--synthetic-n", type=int, default=0,
                        help="rows of synthetic two-hop/numeric decisions to mix in (0 = off)")
    parser.add_argument("--init-checkpoint-s3", type=str, default="",
                        help="S3 prefix of an existing full checkpoint (option_marker.pt + config) "
                             "to continue training from instead of a fresh scoring head.")
    parser.add_argument("--base-encoder-s3", type=str, default="",
                        help="S3 prefix of a plain encoder dir (config/safetensors/tokenizer) to start from with a fresh head")
    parser.add_argument("--independent-options", action="store_true",
                        help="train with the order-invariant independent-option attention mode")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--trainer", default="marker", choices=["marker", "decoder"],
                        help="marker = ModernBERT option-marker (default); decoder = Qwen3.5 + LoRA + JointSchemaHead")
    parser.add_argument("--base-model-id", default="", help="Hub id of the backbone (decoder trainer); default Qwen/Qwen3.5-0.8B")
    parser.add_argument("--decoder-opt", action="append", default=[],
                        help="decoder trainer knob as key=value: max_length, max_tokens, lora_r, lora_alpha, head_width, head_routing_layers, head_layers, head_heads, head_feedforward, eval_name, polarity_weight, polarity_margin")
    parser.add_argument("--extra-rows", type=int, default=0, help="row count of --extra-train-s3, for the watchdog estimate")
    parser.add_argument("--watchdog-min", type=int, default=0,
                        help="hard shutdown ceiling in minutes (0 = derive from epochs)")
    args = parser.parse_args()

    launch(
        on_demand=args.on_demand,
        epochs=args.epochs,
        s3_target=args.s3_target,
        max_train=args.max_train,
        long_context=args.long_context,
        long_ratio=args.long_ratio,
        overlap_target=args.overlap_target,
        synthetic_n=args.synthetic_n,
        extra_train_s3=args.extra_train_s3,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        only_type=args.only_type,
        region=args.region,
        init_checkpoint_s3=args.init_checkpoint_s3,
        base_encoder_s3=args.base_encoder_s3,
        independent_options=args.independent_options,
        lr=args.lr,
        watchdog_min=args.watchdog_min,
        trainer=args.trainer,
        extra_rows=args.extra_rows,
        base_model_id=args.base_model_id or ("Qwen/Qwen3.5-0.8B" if args.trainer == "decoder" else "wfzyx/von"),
        decoder_opts={k: (int(v) if v.isdigit() else float(v) if v.replace(".", "", 1).isdigit() else v) for k, v in (o.split("=", 1) for o in args.decoder_opt)},
    )
