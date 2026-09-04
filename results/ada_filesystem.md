# Ada filesystem and compute inventory (P0.1, P0.3)

Measured on 2 September 2026 by workstream A, from the head node and from
compute nodes gnode012, gnode056, gnode063, gnode078 and gnode092. Everything
below is observed, not quoted from documentation.

This resolves **open decision 8** and fills the `TBD` fields in
`configs/activations.yaml`. It also changes the answer to the layer-quota rule
in that file, so read Section 5 before the evening sync.

## 1. Account access (P0.1)

All four members exist and hold a SLURM association, so nobody is blocked on a
multi-day account request.

| Member | Ada username | uid | SLURM account | QOS |
| --- | --- | --- | --- | --- |
| Yash More | `yash.more` | 4041 | research | medium |
| Naman Singhal | `naman.s` | 4082 | research | medium |
| Shrish Kadam | `shrish.kadam` | 4093 | research | medium |
| Sanjith Ganapathi | `sanjith.ganapathi` | 3906 | research | **low** |

All four share gid 2000 (`research`), which is what makes a group-writable
directory on a node's `/scratch` usable between us.

Two things to note. Sanjith is on QOS `low`, so his jobs queue behind the rest
of ours; his workstream is the synthetic generator and needs no GPU, so this
costs us nothing as long as we do not move GPU work to him. And a per-user CPU
cap (`QOSMaxCpuPerUserLimit`) is enforced: submitting several wide jobs at once
leaves the later ones pending rather than running, which matters when we start
the ChartQA passes.

**Still outstanding:** I verified accounts and associations, which is what
actually gates submission, but I cannot submit a job *as* another member. Each
of the other three should run this once and paste the output:

```bash
srun -p u22 --constraint=2080ti -c 1 --mem=2G -t 00:05:00 \
     bash -c 'hostname; mkdir -p /scratch/vlm-failures && echo SCRATCH_OK; quota -s | tail -3'
```

## 2. What is actually shared

This is the finding that matters, and it is not what SPEC Section 4.1 assumed.

| Path | Visible from | Size / quota | Writable | Durable |
| --- | --- | --- | --- | --- |
| `$HOME` (`/home2/<user>`) | head node **and** every compute node | **30 GiB per user** | yes | yes |
| `/scratch` | **only the node you are on** | 1.8 TB, ~1.5 TB free | yes | **no, purged, see Section 3** |
| `/ssd_scratch` | only the node you are on | 880 GB | yes | no |
| `/share1` | **only the head node** | 100 GiB per user | yes | yes, but unreachable from jobs |
| `/share7`, `/data3`, `/cvit_contrib_packages` | compute nodes | large | **read-only** | n/a |

`$HOME` is the only path that is simultaneously shared, writable and durable.

`/share1` looks like the answer and is not. It is a local disk on the head node
(`/dev/sdg1`) and is not mounted on any compute node. A job cannot write to it
directly, and a compute node cannot reach the head node over ssh either:

```
$ ssh ada.iiit.ac.in            # from gnode063, inside a job
yash.more@ada.iiit.ac.in: Permission denied (publickey,password,hostbased).
```

So the 100 GiB in `/share1` is reachable only by hand, from the head node,
after a job has already put its output somewhere the head node can see. It is
usable as an archive, not as `durable_root`. See Section 5 for the one change
that would unlock it.

**Home quota, measured:** 30720 MiB soft, 31744 MiB hard. `yash.more` is at
18113 MiB, leaving **about 12.6 GiB free** before the project venv. The venv
itself is 6.0 GiB, so roughly **6.5 GiB free today**. The other three members
have their own 30 GiB each and have not been measured.

## 3. `/scratch` is purged, and it is really `/tmp`

`/scratch` and `/tmp` are the same directory, bind-mounted twice:

```
$ stat -c '%d:%i' /scratch /tmp
2065:2
2065:2
$ grep -E ' /(tmp|scratch) ' /proc/mounts
/dev/sdb1 /scratch ext4 rw,relatime 0 0
/dev/sdb1 /tmp     ext4 rw,relatime 0 0
```

A file written to `/scratch/x` appears at `/tmp/x`. That matters because
`tmpreaper` runs daily from `/etc/cron.daily/tmpreaper` over `TMPREAPER_DIRS`,
which is `'/tmp/.'`, and `TMPTIME` is unset in `/etc/default/rcS`, so the
default applies.

**Answer to P0.3(c): `/scratch` is purged daily of anything not accessed in
7 days.** The empirical check agrees: across 129 entries from many users on
gnode063, the oldest surviving entry was 5 days old and nothing older than a
week was present. Separately, `/ssd_scratch/cvit` has its own explicit rule at
`+10d`.

Consequence: the HuggingFace cache on `/scratch` is not something we set up
once. It has to be repaired on demand, which is why `scripts/ada_env.sh`
carries `stage_in_model` and every job calls it. A checkpoint touched at least
weekly survives; one left alone over a quiet fortnight does not.

## 4. GPU inventory, and why the constraint is not optional

Partition `u22` (78 nodes) is our pool. `ihub`, `plafnet2` and `rrc` exist but
belong to other groups.

```
$ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
NVIDIA GeForce RTX 2080 Ti, 7.5, 11264 MiB, 570.211.01
```

Nodes are genuinely mixed, and SLURM exposes the difference as a feature label:

| Feature | GPUs per node | Scheduling weight |
| --- | --- | --- |
| `2080ti,phase3` | 4 | 100 |
| `2080ti` | 4 (some 3) | 100 |
| *(no feature)* | 3 | 200 |

Unlabelled nodes are the older tranche. Since `configs/activations.yaml`
freezes one GPU type for the project, every job carries
`--constraint=2080ti` and `--exclude=gnode077`, and `ada_env.sh` re-checks the
card at runtime via `require_gpu_type` rather than trusting the scheduler.

**Set `gpu_type: 2080ti` in `configs/activations.yaml`.**

Three consequences of Turing (sm_75) that change how the models are run, and
all three are silent failures rather than loud ones:

1. **11 GiB per card, so a 7B model does not fit on one GPU.** Qwen2.5-VL-7B is
   about 8.3 B parameters including the vision tower, which is roughly 16.6 GiB
   in fp16 against 11.26 GiB of VRAM. Every job asks for `--gres=gpu:2` and
   shards with `device_map="auto"`. Quantising to fit one card is not an option
   for this project: we probe internal activations, and quantisation perturbs
   the quantity being measured.
2. **No bf16 tensor cores.** Qwen2.5-VL's reference code runs bf16, which
   Ampere and later support natively. On Turing a bf16 matmul returns finite
   numbers, so nothing crashes, but it is emulated. **The project runs fp16**,
   and that is recorded next to the library versions because it is exactly the
   kind of difference that moves activations without moving accuracy much.
3. **No FlashAttention-2**, which needs sm_80 or later. We use
   `attn_implementation="sdpa"`.

`src/extract/smoke_qwen.py` picks dtype and attention from the observed compute
capability rather than hardcoding them, and refuses to start if less than 17
GiB of VRAM is visible.

## 5. Where `HF_HOME` and `durable_root` point (open decision 8)

**`HF_HOME=/scratch/vlm-failures/hf`.** Node-local, 1.5 TB free, group-writable
so all four of us share one copy per node. It cannot go in `$HOME`: two 7B
checkpoints are roughly 33 GiB combined against a 30 GiB quota with 6.5 GiB
free. Because `/scratch` is purged at 7 days and is per node, `stage_in_model`
in `scripts/ada_env.sh` re-downloads on a cold node and is a no-op on a warm
one. Jobs constrained to `2080ti` keep landing in the same pool, so the cache
is usually warm.

**`durable_root=$HOME/anlp/project/vlm-failures-durable`.** It is the only
shared writable path. Every job stages results there before exiting (P0.9), and
`stage_out` in `ada_env.sh` fails the job loudly if the copy does not happen.

### This changes the layer rule, and Naman needs to know before the sync

`configs/activations.yaml` says:

```
durable_root quota >= 100 GiB  ->  layers: all
durable_root quota <  100 GiB  ->  layers: subsampled
```

The measured quota is **30 GiB total, about 6.5 GiB free on my account today**.
That is far below 100 GiB, so as written **the rule selects the subsampled
fallback, not `layers: all`**, and the 34.5 GiB full cache has nowhere to live.
Even the 9.2 GiB subsampled cache does not fit in my current free space without
clearing something first.

`configs/activations.yaml` is Naman's file and the rule is his to apply, so I
have filled in the measured numbers rather than flipping `layers` myself. Four
ways out, cheapest first:

1. **Shard the cache across the four homes.** 4 x 30 GiB is 120 GiB, which
   clears the 100 GiB bar. Each member owns one shard and the probe training
   step reads all four paths. Costs nothing, available today, and the paths are
   already readable between us.
2. **Free space in home.** I am at 18.1 GiB before the venv, and I have not
   deleted anything. A 6.3 GiB `totto` conda env and a 3.1 GiB pip cache are
   the obvious candidates, but that is Yash's call, not an infra decision.
3. **Set up an ssh key so jobs can rsync to `/share1`.** A key pair in the
   shared `$HOME/.ssh` would make the head node reachable from compute nodes
   and unlock 100 GiB per user, which clears the bar on its own. This is a
   credential change on a personal account, so it is not something to do
   silently; raise it and decide as a team.
4. **Ask the sysadmins for a group share.** Correct long-term answer, multi-day
   turnaround, so start it now and do not wait on it.

Option 1 needs no permission from anyone and is the recommendation for the
evening sync.

## 6. Reproducing this

`scripts/ada_env.sh` centralises the paths and both guard functions.
`scripts/smoke.sbatch` and `scripts/verify_tokens.sbatch` are the two job
templates and both source it.
