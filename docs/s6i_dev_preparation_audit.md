# S6I Dev Preparation Audit

审计日期：2026-09-17（Asia/Shanghai）

范围：MISP 2025 AVSR baseline 的 Dev 远场准备链路、现有服务器数据/工作产物和 baseline scorer。
限制：本审计只读 baseline、原始数据和现有 work 产物；没有执行 preparation、GSS、ASR、训练或大规模 pipeline，没有修改 baseline 或原始数据。

## 1. Executive Summary

- **CONFIRMED**：AVSR baseline 的 Dev 远场链路为 prep_far_avsr.sh Stage 0--4 生成 8ch/重命名输入，再由 prepare_gss_data.py（Stage 2）生成 Kaldi data dir。segments、text、utt2spk 首次同时完整出现在 Stage 2 收尾；spk2utt 随后由 Kaldi 工具生成。最终 wav.scp 被 channels.scp 覆盖，因此两者都指向 recording stem。
- **CONFIRMED**：当前服务器已有 Dev Stage-1 音频和 TextGrid：6 个 session、48 个重命名 8ch wav、30 个重命名 TextGrid；但 /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far 不存在，没有可直接用于 S6I 的 Dev Kaldi data dir。
- **CONFIRMED**：Dev AVSR 实际 authoritative transcription 输入是 F8N TextGrid，不是 JSON。JSON 在 data/dev/transcription 中存在，但 baseline 远场准备链路不解析 JSON；prepare_gss_data.py 只读取重命名 TextGrid 的 内容层。
- **CONFIRMED**：可以只生成 Dev preparation artifacts 而不跑 GSS/ASR，使用仓库现有 scipts/prep_far_avsr_gss.sh 的 --stage 2 --stop-stage 2 --gss-direct。原始 s0/run_asr_far.sh 不适合作为本服务器的安全入口。
- **RECOMMENDED**：首个完整真实 Dev E2E 选 M028，完整处理其 4 个 speaker、144 条 segment，不做人工 subset。recording stem 是 M028_S197199201241_F8N_Far，8 个物理 wav 是 _0.wav--_7.wav。
- **BLOCKER**：还需实际执行一次 Stage 2 才会有 Dev Kaldi dir；随后按 exact utterance ID 做 reconciliation，再运行 S5/S6。Stage 3 GSS 不是 raw_ch0 S6I 的前置条件。

## 2. Current Server State

### 2.1 Source and work paths

**CONFIRMED**：

    /home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/
    /home/gc6_challenge/data/
    /home/gc6_challenge/work/misp_avsr_far/
    /home/gc6_challenge/whisper_asr/

当前 whisper_asr 分支为 feat/whisper-backend-v0；审计开始时工作树无未提交变更。baseline 文件只读审计，本次没有执行会写 baseline 的命令。

### 2.2 Existing Dev preparation

**CONFIRMED**：

| 产物 | 实际状态 |
|---|---:|
| work/misp_avsr_far/dev_far_audio_multi_channel/*.wav | 48 |
| work/misp_avsr_far/dev_far_audio_multi_channel_rename/*.wav | 48 |
| work/misp_avsr_far/dev_transcription_raw/*.TextGrid | 30 |
| work/misp_avsr_far/dev_transcription_rename/*.TextGrid | 30 |
| _gss_sandbox/data/training_far/segments | 29,281 行 |
| _gss_sandbox/data/training_far/text | 29,281 行 |
| _gss_sandbox/data/training_far/wav.scp | 13 行 |
| _gss_sandbox/data/training_far/channels.scp | 13 行 |
| _gss_sandbox/data/training_far/utt2spk | 29,281 行 |
| _gss_sandbox/data/training_far/spk2utt | 58 行 |
| _gss_sandbox/data/dev_far/ | MISSING |

当前未发现 Dev data.list、dev_far_audio_segment、GSS enhanced/*.wav、misp_data 或 dump。Dev Stage-3 enhanced frontend output 尚未生成。

Dev 8ch 完整性（每个 session 均为 8/8）：

    M014_S161164189190191_F8N_Far
    M016_S037213214215216_F8N_Far
    M017_S223224225226227228229230_F8N_Far
    M024_S034037160237_F8N_Far
    M026_S197199201241_F8N_Far
    M028_S197199201241_F8N_Far

现有重命名文件为 RIFF PCM、32-bit、mono、16 kHz。Dev 重命名音频约 5.6G；原始 Dev CSOBx3 和 F8N 音频目录约 5.9G、1.8G。

### 2.3 Raw Dev source

**CONFIRMED**：Dev transcription JSON 共 30 个，Dev F8N TextGrid 共 30 个，work raw/rename TextGrid 也各 30 个。远场 PCM 位于：

    /home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M014/M014-CSOBx3/M014-CSOBx3.pcm
    /home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M016/M016-CSOBx3/M016-CSOBx3.pcm
    /home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M017/M017-CSOBx3/M017-CSOBx3.pcm
    /home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M024/M024-CSOBx3/M024-CSOBx3.pcm
    /home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M026/M026-CSOBx3/M026-CSOBx3.pcm
    /home/gc6_challenge/data/dev/audio/CSOBx3/dev-CSOBx3/M028/M028-CSOBx3/M028-CSOBx3.pcm

F8N TextGrid 的实际目录形式为：

    /home/gc6_challenge/data/dev/audio/F8N/dev-F8N/<Mxxx>/<Mxxx-F8N>/<Mxxx-F8N>-<speaker>.TextGrid

## 3. Dev Preparation Dependency Graph

    data/dev（只读原始树）
      ├─ CSOBx3/*.pcm + timestamp.txt
      │    └─ prep_far_avsr.sh Stage 0 mirror
      │         └─ Stage 1 pcm2wav.py
      │              └─ Stage 2 wav2multi_channel_wav.py
      │                   └─ {set}_far_audio_multi_channel/*.wav
      │                        └─ Stage 4 rename_wav_textgrid.py
      │                             └─ {set}_far_audio_multi_channel_rename/*_{0..7}.wav
      └─ F8N/*.TextGrid + timestamp.txt
           └─ Stage 3 textgrid_cutoff.py
                └─ Stage 4 rename_wav_textgrid.py
                     └─ {set}_transcription_raw/*.TextGrid
                     └─ {set}_transcription_rename/*_Near_<speaker>.TextGrid

    renamed 8ch wav + renamed TextGrid
      └─ Stage 2 prepare_gss_data.py
           ├─ temp/channels.scp
           ├─ temp/wav.scp
           ├─ temp/segments
           ├─ temp/text_sentence
           └─ temp/utt2spk
                └─ sort/uniq + channels.scp → wav.scp + Kaldi tools
                     ├─ segments
                     ├─ text
                     ├─ utt2spk
                     └─ spk2utt

    Stage 3 run_gss.sh（本次不运行）
      └─ manifests/lhotse/GSS enhance/enhanced wav/final segment data

### 3.1 Stage table

| Stage | 脚本、函数、关键行 | 输入 → 输出 | 写文件 | 依赖 |
|---|---|---|---|---|
| 0 | scipts/prep_far_avsr.sh:592-604；mirror_set:500-549 | data/{set} → _sandbox/{set} symlink mirror | 是，仅 work sandbox | raw set |
| 1 | s0/local/pcm2wav.py:21-45 | .pcm → 同树 .wav、conversion log | 是 | Stage 0；numpy/scipy |
| 2 | s0/local/wav2multi_channel_wav.py:22-59；调用于 prep_far_avsr.sh:627-647 | 8ch wav + timestamp → multi_channel/*.wav、segments_paths.txt | 是 | Stage 1 |
| 3 | s0/tools/textgrid_cutoff.py:9-84；调用于 prep_far_avsr.sh:649-665 | F8N TextGrid + timestamp → *-cutoff.TextGrid | 是 | Stage 0 |
| 4 | s0/local/rename_wav_textgrid.py:7-88；调用于 prep_far_avsr.sh:668-692 | multi-channel wav/TextGrid → Far wav、Near TextGrid | 是，shutil.copy2 | Stage 2+3 |
| 5 | s0/gss_main/prepare_gss_data.py:261-289,306-376,407-448 | renamed inputs → store/temp manifests | 是 | Stage 4；Python packages |
| 6 | s0/local/prepare_gss_data.sh:34-54；wrapper 等价实现 :961-987 | temp → final Kaldi files | 是 | Kaldi utils |
| 7 | s0/local/run_gss.sh:22-103 | Stage 2 data → GSS manifests/enhanced/final data | 是，且很大 | Stage 6；lhotse/GSS/GPU |

### 3.2 First complete appearance

**CONFIRMED**：

1. prepare_gss_data.py:273 写 temp/channels.scp；
2. :401-403 写 temp/segments、temp/text_sentence、temp/utt2spk；
3. prep_far_avsr_gss.sh:966-979（等价于 prepare_gss_data.sh:34-45）写最终 segments、text、utt2spk、channels.scp、wav.scp；
4. prepare_gss_data.sh:49-54 / wrapper :981-987 生成 spk2utt 和 fix_data_dir。

所以 Dev Kaldi artifacts 首次完整出现于 **Stage 2 完成后**，不是 Stage 3。

### 3.3 Native support judgment

**CONFIRMED（分入口）**：

- 阶段划分本身支持：run_asr_far.sh:88-96 是 Stage 2，:102-108 是 Stage 3。
- 原始 entrypoint 在当前服务器上不安全：它 source prepare_gss_data.sh，而该脚本 :59 是 exit 0；run_asr_far.sh:51-57 还是作者机器绝对路径。另有 prepare_gss_data.sh:26/29 的 _opt/_opts 拼写问题，导致 --channel_dir 丢失；worker 随后在 :320 读取不存在的 temp/channels.scp。
- 仓库已有 scipts/prep_far_avsr_gss.sh：--gss-direct 直接调用 baseline prepare_gss_data.py，并在 _gss_sandbox 做原 Stage 2 收尾；--stage 2 --stop-stage 2 不进入 run_gss.sh。

结论：**YES**，用现有 wrapper Stage 2-only 可以不跑 GSS/ASR；**NO**，不要直接使用原始 run_asr_far.sh 作为当前服务器安全入口。

## 4. Artifact Contract

### 4.1 Kaldi 文件格式

**CONFIRMED**（prepare_gss_data.py:261-289,303-374）：

    segments: <utt-id> <recording-id> <begin-sec> <end-sec>
    text:     <utt-id> <reference text>
    wav.scp:  <recording-id> <recording stem/prefix>
    channels.scp: <recording-id> <recording stem/prefix>
    utt2spk:  <utt-id> S<speaker>
    spk2utt:  <speaker> <utt-id> ...

prepare_channels_scp 对 *_0.wav--*_7.wav 做 path[:-6]，去掉 _0.wav 六个字符后去重；所以每个 recording 只写一条 stem。最终 shell 收尾在 prepare_gss_data.sh:45 以 channels.scp 覆盖 wav.scp。

### 4.2 segments/text/ID 规则

**CONFIRMED**（prepare_gss_data.py:310-375）：

- 只读取 tier 内容层；角色层不进入 AVSR text；
- rejected list 包括 <NOISE>、<knock>、<其他说话人> 等，且 duration 必须至少 0.12 秒；
- start 向下、end 向上对齐到 0.04 秒；
- 去掉 !！。.,，?？、 和空格，再排除 呃/啊/噢/嗯/唉/<NOISE>/**；
- utterance ID 是 S{speaker}_{room}_{speakers}_{config}_{start_ms:06d}-{end_ms:06d}；
- segments recording-id 是同一 recording stem，utt2spk 是 S{speaker}。

### 4.3 Stem resolution

**CONFIRMED**：s0/gss_main/prepare_misp.py:92-116,134-151 读取 wav.scp 的 stem，计算 stem_{0..7}.wav duration，并为 channel_id=0..7 构造物理 source。wav.scp 第二列不是单个 wav 是 baseline 的多通道 contract；S6I raw_ch0 应用同一规则解析 stem + _0.wav。

## 5. One Real Dev Example

**CONFIRMED**，M028：

| 项 | 实际值 |
|---|---|
| 原始 transcription | /home/gc6_challenge/data/dev/transcription/M028/M028-Transcription/M028-Transcription-197.json |
| baseline 实际读取 TextGrid | /home/gc6_challenge/data/dev/audio/F8N/dev-F8N/M028/M028-F8N/M028-F8N-197.TextGrid |
| session | M028 |
| speaker set | 197, 199, 201, 241 |
| recording ID/stem | M028_S197199201241_F8N_Far |
| physical channels | .../M028_S197199201241_F8N_Far_0.wav 到 _7.wav |
| segment ID format | S197_M028_S197199201241_F8N_<start-ms>-<end-ms> |

TextGrid 第一 tier 是 内容层，xmax=798.277。一个真实 accepted interval 是 xmin=141.53、xmax=150.75 的中文内容；按 baseline 对齐规则预期 ID 为：

    S197_M028_S197199201241_F8N_014152-015076

prepared reference 文件是：

    /home/gc6_challenge/work/misp_avsr_far/dev_transcription_rename/M028_S197199201241_F8N_Near_197.TextGrid

JSON 含 content、CSOBx3 duration、F8N duration，但本 baseline AVSR preparation 没有 JSON parser。**CONFIRMED**：此链路不需要从 JSON 再转换一次；使用已存在的 renamed TextGrid。
**UNRESOLVED（旁支）**：data/eval/transcription 当前为空而 work 中有 eval TextGrid；不据此推断 Dev。

## 6. Recommended Full Dev Session

计数使用 baseline parser/filter 的只读等价审计：内容层、rejected list、0.12 秒门槛、punctuation/sound filter，无人工 subset。

| session / recording stem | TextGrid speakers | segments | 8ch |
|---|---:|---:|---|
| M014_S161164189190191_F8N_Far | 5 | 562 | 8/8 |
| M016_S037213214215216_F8N_Far | 5 | 309 | 8/8 |
| M017_S223224225226227228229230_F8N_Far | 8 | 419 | 8/8 |
| M024_S034037160237_F8N_Far | 4 | 555 | 8/8 |
| M026_S197199201241_F8N_Far | 4 | 529 | 8/8 |
| M028_S197199201241_F8N_Far | 4 | 144 | 8/8 |
| Dev total | 30 | 2,518 | 48/48 |

**RECOMMENDED：M028，144 segments。** 它是完整 Dev session 中最小者，4 个 speaker 的 TextGrid 完整、8 个物理通道完整、命名符合所有其他 Dev/training session，没有发现人工修补需求；不是人工截取 subset。其音频路径为：

    /home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/M028_S197199201241_F8N_Far_{0..7}.wav

M016 是次小完整 session（309 segments），可作为第二个中等长度 sanity case。

## 7. Minimal Safe Preparation Command

以下命令是建议，**本次没有执行**：

    cd /home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main
    bash scipts/prep_far_avsr_gss.sh \
      --data /home/gc6_challenge/data \
      --out /home/gc6_challenge/work/misp_avsr_far \
      --sets dev \
      --meetings M028 \
      --stage 2 --stop-stage 2 \
      --gss-direct \
      --keep-sandbox \
      --yes

**CONFIRMED（wrapper 源码）**：

- 两个 stage 参数必须同时给；只给 --stage 2 会继续 Stage 3（wrapper:138-143,197-203）。
- --gss-direct 调 baseline prepare_gss_data.py（:948-959），不调用 run_gss.sh（:1010-1025）。
- --meetings M028 建立 _gss_input/dev 的 symlink view，不复制 raw audio。
- 已存在的 Stage-1 renamed audio/TextGrid 是输入，无需再跑 Stage 1。

预计写入：

    /home/gc6_challenge/work/misp_avsr_far/_gss_input/dev/{wav,textgrid}
    /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/
      temp/ channels.scp wav.scp segments text utt2spk spk2utt .gc6_provenance

不应写 data/dev、training_far、data_far_audio/dev_far_audio_segment、misp_data、dump 或 enhanced。Stage 2 manifest/text 预计 MB 级；M028 已有 renamed 8ch 音频约 408,729,440 bytes，本命令不应再次复制这 409MB。

如需整个 Dev split，删除 --meetings M028；预期 2,518 条 segment。不要加 --replace，也不要为 S6I 重新跑 Stage 1。wrapper:92-101 已说明 scoped store_dir 会覆盖该 set 的 Stage 2 文件，Stage 3 旧输出不会自动清除。

## 8. Baseline Scorer Parity Contract

### 8.1 Actual call

**CONFIRMED**：s0/run_asr_far.sh:230-231：

    python tools/compute-wer.py --char=1 --v=1 ${eval_text} $test_dir/text > $test_dir/wer

本文称它为 **MISP AVSR baseline scorer**，不是独立官方 challenge submission evaluator；代码没有证明更高 authority。

### 8.2 Input and ID matching

compute-wer.py:353-386 和 compute-cer.py:372-408 要求两个 UTF-8 文本，每行：

    <utterance-id> <payload>

第一参数 ref、第二参数 hyp。hyp 先读成 fid -> tokens 字典，ref 按行只计算 hyp 中存在的 ID。因此不要求排序，但必须 exact ID；hyp 重复 ID 后写覆盖；missing/extra 不进入 overall。S6I 必须在 scorer 前显式检查 missing/extra/duplicate，不能依赖 scorer 的交集语义。

### 8.3 Normalization

**CONFIRMED，compute-wer.py:14-79**：

- --char=1 删除代码列出的中英文 punctuation 和空白；中文 Lo 字符逐字切分，ASCII/数字/标签等可能保留为一个 token；
- 默认 --cs=0 时 uppercase；
- remove_tag=True，normalize 删除 <...> tag；
- 没有通用 Unicode punctuation normalization，只处理源码 puncts 列表；
- 不用 jieba 分词；
- 这套 scorer normalization 独立于 prepare_gss_data.py:312,368-370 的 text 清理。

### 8.4 compute-wer vs compute-cer

**CONFIRMED**：compute-wer.py:14-44 的 char mode 对中文逐字但对非中文内容可能按 token；compute-cer.py:15-46,65-86 在 normalize 中对 x.isalnum() 再逐字符展开。因此英文/数字的行为不同。两者都输出：

    Overall -> ... N=... C=... S=... D=... I=...

字段可与 S5 的 S/D/I/N/CER 对齐，前提是同一 ref/hyp ID 集合和同一脚本。baseline 实际以 compute-wer.py --char=1 为准。

### 8.5 Future M028 parity command template

    cd /home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main
    python s0/tools/compute-wer.py --char=1 --v=1 \
      /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/text \
      /path/to/m028_hyp.text \
      > /path/to/m028_baseline_wer

这只是模板，本次未执行；dev_far/text 只有 Stage 2 后才存在。

## 9. Risks / Blockers

1. **CONFIRMED**：不要对 /home/gc6_challenge/data/dev 直接执行原脚本；prep_far_avsr.sh 的安全工作版本通过 sandbox mirror，pcm2wav 写 sandbox。
2. **CONFIRMED**：原 run_asr_far.sh:91-95 的 store_dir 写 baseline s0/gss_main；Stage 3 还会写 misp_data/dump。wrapper 才把它们放进 _gss_sandbox。
3. **CONFIRMED**：prepare_gss_data.sh:59 的 source/exit 耦合。
4. **CONFIRMED**：prepare_gss_data.sh:26,29-30 的 _opt/_opts bug；使用 --gss-direct。
5. **CONFIRMED**：Stage 参数误用可能自动启动 GSS；必须 --stage 2 --stop-stage 2。
6. **CONFIRMED**：全 Dev renamed 8ch wav 约 5.6G；重跑 Stage 1 会产生中间 8ch、8 个 mono 和 renamed copy。Stage 2-only 不应生成音频副本。
7. **CONFIRMED**：run_gss.sh:50-71 的 enhance 使用 CUDA_VISIBLE_DEVICES=3 和 --force-overwrite；Stage 2-only 不触发且不需 GPU。
8. **CONFIRMED**：实际目录拼写是 scipts/，不是 scripts/。
9. **CONFIRMED**：Stage 1/2 需要现有 Python 环境能提供 numpy/scipy/soundfile/tqdm/jieba/zhon；Stage 3 另需 lhotse/GSS。wrapper 的 lib_guard 会选择/检查解释器；本次没有启动 pipeline 验证环境。
10. **CONFIRMED**：raw_ch0 S6I 不依赖 GSS enhanced output。
11. **UNRESOLVED**：eval 历史 TextGrid provenance；本次不重建 eval。

## 10. Exact Next Step for S6I

1. 执行前只读确认 dev_far 尚不存在，M028 有 8/8 wav 和 4 个 renamed TextGrid。
2. 执行第 7.1 节 Stage-2-only 命令；不加 --replace，不执行 run_asr_far.sh 或 run_gss.sh。
3. 验证 M028：segments=text=utt2spk=144，wav.scp=channels.scp=1，spk2utt=4；检查 recording-id、duplicate ID。
4. 用 stem + _0.wav 进入现有 S6I adapter，保留 segments 的 start/end 和 utterance ID。
5. 跑 Whisper large-v3、S5/S6；随后用第 8.5 节 baseline scorer 做 parity，先比 reconciliation 再比 S/D/I/N/CER。
6. M028 全部 reference/hyp exact reconciliation 完成后，作为 S7 前真实 Dev E2E evidence。

## 11. Evidence Appendix

### Baseline source evidence

- /home/gc6_challenge/code/AVSR/MISP2025-AVSR-baseline-main/scipts/prep_far_avsr.sh:592-705：Stage 0 mirror、Stage 1 PCM conversion、Stage 2 split、Stage 3 cutoff、Stage 4 rename。
- s0/local/pcm2wav.py:21-45：int32 bytes reshape 为 (-1,8)，写 16000 Hz WAV，不 resample。
- s0/local/wav2multi_channel_wav.py:22-59：timestamp crop、按 ch 写 wav、写 segments_paths.txt。
- s0/tools/textgrid_cutoff.py:9-84：timestamp cutoff；无 timestamp 时复制。
- s0/local/rename_wav_textgrid.py:41-74：按 meeting 分组、拼 speaker suffix、shutil.copy2。
- s0/gss_main/prepare_gss_data.py:261-289：scp；:303-376：filter/time/ID/segments/text/utt2spk；:379-404：temp；:407-448：CLI。
- s0/local/prepare_gss_data.sh:25-30,34-54,58-59：bug、收尾、spk2utt/fix_data_dir、exit。
- s0/run_asr_far.sh:50-57,74-109：绝对路径、Stage 1--3；:230-231：scorer。
- scipts/prep_far_avsr_gss.sh:109-143,279-303,888-1008,1010-1055：sandbox、meeting view、Stage 2/3 分界。
- s0/gss_main/prepare_misp.py:92-116,134-151：stem 到 _0.wav--_7.wav。
- s0/tools/compute-wer.py:14-79,353-386,452-459 与 s0/tools/compute-cer.py:15-86,372-408,477-486：输入、normalization、ID 和统计。

### Server read-only evidence

    find /home/gc6_challenge/data/dev/transcription -type f -name '*.json' | wc -l
    30
    find /home/gc6_challenge/data/dev/audio/F8N -type f -name '*.TextGrid' | wc -l
    30
    find /home/gc6_challenge/work/misp_avsr_far/dev_transcription_raw -type f -name '*.TextGrid' | wc -l
    30
    find /home/gc6_challenge/work/misp_avsr_far/dev_transcription_rename -type f -name '*.TextGrid' | wc -l
    30
    find /home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename -type f -name '*.wav' | wc -l
    48

    /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/training_far/segments  29281 lines
    /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/training_far/text      29281 lines
    .../training_far/wav.scp                                                       13 lines
    .../training_far/channels.scp                                                  13 lines
    .../training_far/utt2spk                                                       29281 lines
    .../training_far/spk2utt                                                       58 lines
    /home/gc6_challenge/work/misp_avsr_far/_gss_sandbox/data/dev_far/             MISSING

    /home/gc6_challenge/data/dev/audio/F8N/dev-F8N/M028/M028-F8N/M028-F8N-197.TextGrid  29256 bytes
    /home/gc6_challenge/work/misp_avsr_far/dev_far_audio_multi_channel_rename/M028_S197199201241_F8N_Far_0.wav  51091180 bytes
    file: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 32 bit, mono 16000 Hz

    du -sh:
    data/dev/audio/CSOBx3                                      5.9G
    data/dev/audio/F8N                                         1.8G
    work/misp_avsr_far/dev_far_audio_multi_channel_rename       5.6G
    work/misp_avsr_far/_gss_sandbox                             24M

### Classification

- **CONFIRMED**：由现有文件内容、代码行、只读计数或现有路径直接证明。
- **INFERRED**：下一步操作的合理推导，例如 M028 作为首个 E2E、Stage 2 manifest 为 MB 级；执行前按第 10 节复核。
- **UNRESOLVED**：服务器和代码无法证明的旁支 provenance 或尚未生成的 Dev artifact；没有用猜测填补。
